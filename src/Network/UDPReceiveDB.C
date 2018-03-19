/***************************************************************************
 *
 *   Copyright (C) 2015 Andrew Jameson
 *   Licensed under the Academic Free License version 2.1
 *
 ***************************************************************************/

#include "config.h"

#ifdef HAVE_HWLOC
#include "spip/HardwareAffinity.h"
#endif

#ifdef HAVE_VMA
#include "spip/UDPSocketReceiveVMA.h"
#endif

#include "spip/TCPSocketServer.h"
#include "spip/AsciiHeader.h"
#include "spip/UDPReceiveDB.h"
#include "spip/Time.h"
#include "sys/time.h"

#include <unistd.h>
#include <cstring>
#include <iostream>
#include <stdexcept>
#include <new>
#include <pthread.h>

//#define _DEBUG

using namespace std;

spip::UDPReceiveDB::UDPReceiveDB(const char * key_string)
{
  db = new DataBlockWrite (key_string);

  db->connect();
  db->lock();
  db->page();

  format = NULL;
  control_port = -1;

  control_cmd = None;
  control_state = Idle;

  pthread_cond_init( &cond, NULL);
  pthread_mutex_init( &mutex, NULL);
  verbose = false;

}

spip::UDPReceiveDB::~UDPReceiveDB()
{
  db->unlock();
  db->disconnect();
  delete db;

  if (format)
  {
    format->conclude();
    delete format;
  };
}

int spip::UDPReceiveDB::configure (const char * config_str)
{
  if (verbose)
    cerr << "spip::UDPReceiveDB::configure config.load_from_str(config_str)" << endl;
  // save the header for use on the first open block
  config.load_from_str (config_str);

  if (config.get ("NCHAN", "%u", &nchan) != 1)
    throw invalid_argument ("NCHAN did not exist in config");

  if (config.get ("NBIT", "%u", &nbit) != 1)
    throw invalid_argument ("NBIT did not exist in config");

  if (config.get ("NPOL", "%u", &npol) != 1)
    throw invalid_argument ("NPOL did not exist in config");

  if (config.get ("NDIM", "%u", &ndim) != 1)
    throw invalid_argument ("NDIM did not exist in config");

  if (config.get ("TSAMP", "%f", &tsamp) != 1)
    throw invalid_argument ("TSAMP did not exist in config");

  if (config.get ("BW", "%f", &bw) != 1)
    throw invalid_argument ("BW did not exist in config");

  char * buffer = (char *) malloc (128);

  if (config.get ("DATA_HOST", "%s", buffer) != 1)
    throw invalid_argument ("DATA_HOST did not exist in config");
  data_host = string (buffer);

  if (config.get ("DATA_PORT", "%d", &data_port) != 1)
    throw invalid_argument ("DATA_PORT did not exist in config");

  if (config.get ("DATA_MCAST", "%s", buffer) != 1)
    data_mcast = string ();
  else
    data_mcast = string (buffer);
  if (config.get ("DATA_PORT", "%d", &data_port) != 1)
    throw invalid_argument ("DATA_PORT did not exist in config");


  free (buffer);

  bits_per_second  = (nchan * npol * ndim * nbit * 1000000) / tsamp;
  bytes_per_second = bits_per_second / 8;

  if (verbose)
    cerr << "spip::UDPReceiveDB::configure bytes_per_second=" << bytes_per_second << endl;

  // configure the format based on the config
  if (!format)
    throw runtime_error ("format was not allocated");
  if (verbose)
    cerr << "spip::UDPReceiveDB::configure format->configure()" << endl;
  format->configure (config, "");

  // now write new params to config
  uint64_t resolution = format->get_resolution();
  if (config.set("RESOLUTION", "%lu", resolution) < 0)
    throw invalid_argument ("failed to write RESOLUTION to config");
  if (verbose)
    cerr << "spip::UDPReceiveDB::configure resolution=" << resolution << endl;

  if (db->get_data_bufsz() % resolution != 0)
  {
    cerr << "db->get_data_bufsz=" << db->get_data_bufsz () << endl;
    cerr << "RESOLUTION=" << resolution << endl;
    throw invalid_argument ("Data block buffer size must be multiple of RESOLUTION");
  }

  stats = new UDPStats (format->get_header_size(), format->get_data_size());
  return 0;
}

void spip::UDPReceiveDB::set_format (spip::UDPFormat * fmt)
{
  if (format)
    delete format;
  format = fmt;
}

void spip::UDPReceiveDB::start_control_thread (int port)
{
  control_port = port;
  pthread_create (&control_thread_id, 0, control_thread_wrapper, this);
}

void spip::UDPReceiveDB::stop_control_thread ()
{
  set_control_cmd (Quit);
}

void spip::UDPReceiveDB::set_control_cmd (spip::ControlCmd cmd)
{
  pthread_mutex_lock (&mutex);
  control_cmd = cmd;
  if (control_cmd == Stop || control_cmd == Quit)
    spip::UDPSocketReceive::keep_receiving = false;
  pthread_cond_signal (&cond);
  pthread_mutex_unlock (&mutex);
}

// start a control thread that will receive commands from the TCS/LMC
void spip::UDPReceiveDB::control_thread()
{
  if (control_port < 0)
  {
    cerr << "WARNING: no control port, using 32132" << endl;
    control_port = 32132;
  }

#ifdef _DEBUG
  cerr << "spip::UDPReceiveDB::control_thread creating TCPSocketServer" << endl;
#endif
  spip::TCPSocketServer * control_sock = new spip::TCPSocketServer();

  // open a listen sock on all interfaces for the control port
  if (verbose)
    cerr << "spip::UDPReceiveDB::control_thread open socket on port=" 
         << control_port << endl;
  control_sock->open ("any", control_port, 1);

  int fd = -1;

  char * cmd  = (char *) malloc (32);

  // wait for a connection
  while (control_cmd != Quit && fd < 0)
  {
    // accept with a 1 second timeout
    fd = control_sock->accept_client (1);
    if (fd >= 0 )
    {
      string received = control_sock->read_client (DADA_DEFAULT_HEADER_SIZE);
      const char * cmds = received.c_str();
      control_sock->close_client();
      fd = -1;

      // now check command in list of header commands
      if (spip::AsciiHeader::header_get (cmds, "COMMAND", "%s", cmd) != 1)
        throw invalid_argument ("COMMAND did not exist in header");
      if (verbose)
        cerr << "control_thread: cmd=" << cmd << endl;
      if (strcmp (cmd, "START") == 0)
      {
        // append cmds to header
        header.clone (config);
        header.append_from_str (cmds);
        if (header.del ("COMMAND") < 0)
          throw runtime_error ("Could not remove COMMAND from header");

        if (verbose)
          cerr << "control_thread: open()" << endl;
        open ();

        // write header
        if (verbose)
          cerr << "control_thread: control_cmd = Start" << endl;
        set_control_cmd(Start);
      }
      else if (strcmp (cmd, "STOP") == 0)
      {
        if (verbose)
          cerr << "control_thread: control_cmd = Stop" << endl;
        set_control_cmd(Stop);
      }
      else if (strcmp (cmd, "QUIT") == 0)
      {
        if (verbose)
          cerr << "control_thread: control_cmd = Quit" << endl;
        set_control_cmd(Quit);
      }
    }
  }

#ifdef _DEBUG
  cerr << "spip::UDPReceiveDB::control_thread exiting" << endl;
#endif
}


void spip::UDPReceiveDB::start_stats_thread ()
{
  pthread_create (&stats_thread_id, NULL, stats_thread_wrapper, this);
}

void spip::UDPReceiveDB::stop_stats_thread ()
{
  set_control_cmd(Quit);
  void * result;
  pthread_join (stats_thread_id, &result);
}

/* 
 *  Thread to print simple capture statistics
 */
void spip::UDPReceiveDB::stats_thread()
{
  uint64_t b_recv_total, b_recv_curr, b_recv_1sec;
  uint64_t b_drop_total, b_drop_curr, b_drop_1sec;
  double gb_recv_ps, gb_drop_ps, gb_drop;

  if (verbose)
    cerr << "spip::UDPReceiveDB::stats_thread starting polling" << endl;

  while (control_cmd != Quit)
  {
    b_recv_total = 0;
    b_drop_total = 0;

#ifdef _DEBUG
    cerr << "spip::UDPReceiveDB::stats_thread control_cmd != Quit" << endl;
#endif

    while (control_state == Active)
    {
      // get a snapshot of the data as quickly as possible
      b_recv_curr = stats->get_data_transmitted();
      b_drop_curr = stats->get_data_dropped();

      // calc the values for the last second
      b_recv_1sec = b_recv_curr - b_recv_total;
      b_drop_1sec = b_drop_curr - b_drop_total;

      // update the totals
      b_recv_total = b_recv_curr;
      b_drop_total = b_drop_curr;

      // calculate current rate
      gb_recv_ps = double(b_recv_1sec * 8) / 1000000000;
      gb_drop_ps = double(b_drop_1sec * 8) / 1000000000;
      gb_drop = double(b_drop_total * 8) / 1000000000;

      fprintf (stderr,"Recv %6.3lf [Gb/s] Dropped %6.3lf [Gb/s] %lf [Gb] \n", 
               gb_recv_ps, gb_drop_ps, gb_drop);
      sleep (1);
    }
    sleep(1);
  }
}

bool spip::UDPReceiveDB::open ()
{
  if (verbose > 1)
    cerr << "spip::UDPReceiveDB::open()" << endl;
  
  if (control_cmd == Stop)
  {
    return false;
  }

  if (header.get_header_length() == 0)
  {
    if (verbose)
      cerr << "spip::UDPReceiveDB::open header.clone(config)" << endl;
    header.clone(config);
  }

  // check if UTC_START has been set
  char * buffer = (char *) malloc (128);
  if (header.get ("UTC_START", "%s", buffer) == -1)
  {
    cerr << "spip::UDPReceiveDB::open no UTC_START in header" << endl;
    time_t now = time(0);
    spip::Time utc_start (now);
    utc_start.add_seconds (2);
    std::string utc_str = utc_start.get_gmtime();
    cerr << "Generated UTC_START=" << utc_str  << endl;
    if (header.set ("UTC_START", "%s", utc_str.c_str()) < 0)
      throw invalid_argument ("failed to write UTC_START to header");
  }
  else
    cerr << "spip::UDPReceiveDB::open UTC_START=" << buffer << endl;

  uint64_t obs_offset;
  if (header.get("OBS_OFFSET", "%lu", &obs_offset) == -1)
  {
    obs_offset = 0;
    if (header.set ("OBS_OFFSET", "%lu", obs_offset) < 0)
      throw invalid_argument ("failed to write OBS_OFFSET=0 to header");
  }

  if (header.get ("SOURCE", "%s", buffer) == -1)
  {
    cerr << "spip::UDPReceiveDB::open no SOURCE in header, using J047-4715" << endl;
    if (header.set ("SOURCE", "%s", "J0437-4715") < 0)
      throw invalid_argument ("failed to write SOURCE to header");
  }

  cerr << "spip::UDPReceiveDB::open format->prepare(header, )" << endl;
  format->prepare(header, "");

  open (header.raw());
  free (buffer);

  return true;
}

// write the ascii header to the datablock, then
void spip::UDPReceiveDB::open (const char * header_str)
{
  // open the data block for writing  
  if (verbose)
    cerr << "spip::UDPReceiveDB::open db->open()" << endl;
  db->open();

  // write the header
  if (verbose)
    cerr << "spip::UDPReceiveDB::open db->write_header()" << endl;
  db->write_header (header_str);
}

void spip::UDPReceiveDB::close ()
{
  if (verbose)
    cerr << "spip::UDPReceiveDB::close()" << endl;

  if (db->is_block_open())
  {
    cerr << "spip::UDPReceiveDB::close db->close_block(" << db->get_data_bufsz() << ")" << endl;
    db->close_block(db->get_data_bufsz());
  }

  // close the data block, ending the observation
  if (verbose)
    cerr << "spip::UDPReceiveDB::close db->close" << endl;
  db->close();

  // clear the header
  if (verbose)
    cerr << "spip::UDPReceiveDB::close header.reset()" << endl;
  header.reset();

  // reset the UDP stats
  if (verbose)
    cerr << "spip::UDPReceiveDB::close stats->reset()" << endl;
  stats->reset();
}

// receive UDP packets for the specified time at the specified data rate
bool spip::UDPReceiveDB::receive (int core)
{
  if (verbose)
    cerr << "spip::UDPReceiveDB::receive ()" << endl;

#ifdef HAVE_HWLOC
  spip::HardwareAffinity hw_affinity;
  if (core >= 0)
  {
    if (verbose)
      cerr << "spip::UDPReceiveDB::receive hw_affinity.bind_thread_to_cpu_core(" << core << ")" << endl;
    hw_affinity.bind_thread_to_cpu_core (core);
    hw_affinity.bind_to_memory (core);
  }
#endif

  if (verbose)
    cerr << "spip::UDPReceiveDB::receive creating socket" << endl;
  // open socket within the context of this thread 
#ifdef HAVE_VMA
  UDPSocketReceiveVMA * sock = new UDPSocketReceiveVMA;
#else
  UDPSocketReceive * sock = new UDPSocketReceive;
#endif

  if (data_mcast.size() > 0)
  {
    if (verbose)
      cerr << "spip::UDPReceiveDB::receive opening multicast socket" << endl;
    sock->open_multicast (data_host, data_mcast, data_port);
  }
  else
  {
    if (verbose)
      cerr << "spip::UDPReceiveDB::receive opening socket" << endl;
    sock->open (data_host, data_port);
  }
  
  size_t sock_bufsz = format->get_header_size() + format->get_data_size();
  if (verbose)
    cerr << "spip::UDPReceiveDB::receive sock->resize(" << sock_bufsz << ")" << endl;
  sock->resize (sock_bufsz);
  size_t kernel_bufsz = 64*1024*1024;
  if (verbose)
    cerr << "spip::UDPReceiveDB::receive sock->resize_kernel_buffer(" << kernel_bufsz << ")" << endl;
  sock->resize_kernel_buffer (kernel_bufsz);

  // block accounting
  const int64_t data_bufsz = db->get_data_bufsz();
  int64_t curr_byte_offset = 0;
  int64_t next_byte_offset = 0;
  int64_t overflow_maxbyte = 0;

  // overflow buffer [2 heaps]
  const int64_t overflow_bufsz = 2097152 * 2;
  int64_t overflowed_bytes = 0;
  int64_t overflow_lastbyte = 0;

  char * overflow = (char *) malloc(overflow_bufsz);
  memset (overflow, 0, overflow_bufsz);

  // loop administration
  char * block = NULL;
  bool filled_this_block = false;
  int got;
  int64_t bytes_this_buf = 0;
  int64_t byte_offset;
  unsigned bytes_received;

  control_state = Idle;
  spip::UDPSocketReceive::keep_receiving = true;

  // wait for the starting command from the control_thread
  pthread_mutex_lock (&mutex);
  while (control_cmd == None)
    pthread_cond_wait (&cond, &mutex);
  pthread_mutex_unlock (&mutex);

  // check the new command
  if (control_cmd == Start)
  {
    control_state = Active;
  }

  // data acquisition loop
  while (control_state == Active)
  {
    // open a new data block buffer if necessary
    if (!db->is_block_open())
    {
      block = (char *) db->open_block();
      filled_this_block = false;

      // copy any data from the overflow into the new block
      if (overflow_lastbyte > 0)
      {
        memcpy (block, overflow, overflow_lastbyte);
        overflow_lastbyte = 0;
        bytes_this_buf = overflowed_bytes;
        stats->increment_bytes (overflowed_bytes);
        overflowed_bytes = 0;
      }
      else
        bytes_this_buf = 0;

      // increment the offsets
      curr_byte_offset = next_byte_offset;
      next_byte_offset += data_bufsz;
      overflow_maxbyte = next_byte_offset + overflow_bufsz;

#ifdef _DEBUG
      cerr << "spip::UDPReceiveDB::receive filling buffer "
         << curr_byte_offset << " - " << next_byte_offset
         << " - " << overflow_maxbyte << "] overflow=" << overflow_lastbyte
         << " overflow_bufsz=" << overflow_bufsz << endl;
#endif
    }

    // tighter loop to fill this buffer
    while (!filled_this_block && sock->still_receiving())
    {
      // get a packet from the socket
      got = sock->recv_from();

      if (got == 0)
      {
        set_control_cmd (Stop);
      }
      else
      {
        byte_offset = format->decode_packet (sock->buf_ptr, &bytes_received);

        // ignore if byte_offset is -ve
        if (byte_offset < 0)
        {
          sock->consume_packet();

          // data stream is finished
          if (byte_offset == -2)
          {
            set_control_cmd(Stop);
          }
        }
        // packet that is part of this observation
        else
        {
          // packet belongs in current buffer
          if ((byte_offset >= curr_byte_offset) && (byte_offset < next_byte_offset))
          {
            bytes_this_buf += bytes_received;
            stats->increment_bytes (bytes_received);
            format->insert_last_packet (block + (byte_offset - curr_byte_offset));
            sock->consume_packet();
          }
          // packet belongs in overflow buffer
          else if ((byte_offset >= next_byte_offset) && (byte_offset < overflow_maxbyte))
          {
            format->insert_last_packet (overflow + (byte_offset - next_byte_offset));
            overflow_lastbyte = std::max((byte_offset - next_byte_offset) + bytes_received, overflow_lastbyte);
            overflowed_bytes += bytes_received;
            sock->consume_packet();
          }
          // ignore packets the preceed this buffer
          else if (byte_offset < curr_byte_offset)
          {
            sock->consume_packet();
          }
          // packet is beyond overflow buffer
          else
          {
            filled_this_block = true;
          }
        }

        // close open data block buffer if is is now full
        if (bytes_this_buf >= data_bufsz || filled_this_block)
        {
#ifdef _DEBUG
          cerr << bytes_this_buf << " / " << data_bufsz << " => " <<
                ((float) bytes_this_buf / (float) data_bufsz) * 100 << endl;
          cerr << "spip::UDPReceiveDB::receive close_block bytes_this_buf="
               << bytes_this_buf << " bytes_per_buf=" << data_bufsz << endl;
#endif
          stats->dropped_bytes (data_bufsz - bytes_this_buf);
          db->close_block (data_bufsz);
          filled_this_block = true;
        }
      }
    }
      
#ifndef HAVE_VMA
    // update stats for any sleeps
    uint64_t nsleeps = sock->process_sleeps();
    stats->sleeps(nsleeps);
#endif

    // check for updated control_cmds
    if (control_cmd == Stop || control_cmd == Quit)
    {
#ifdef _DEBUG
      cerr << "spip::UDPReceiveDB::receive control_cmd=" << control_cmd 
           << endl; 
#endif
      if (verbose)
        cerr << "Stopping acquisition" << endl;
      spip::UDPSocketReceive::keep_receiving = false;
      control_state = Idle;
      set_control_cmd (None);
    }
  }

#ifdef _DEBUG
  cerr << "spip::UDPReceiveDB::receive exiting" << endl;
#endif

  // close the data block
  if (verbose)
    cerr << "spip::UDPReceiveDB::receive this->close()"  << endl;
  close();

  if (control_state == Idle)
    return true;
  else
    return false;
}
