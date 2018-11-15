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
#include "spip/Error.h"
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

  data_block_bufsz = db->get_data_bufsz();
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
  // save the header for use on the first open block
  if (verbose)
    cerr << "spip::UDPReceiveDB::configure config.load_from_str(config_str)" << endl;
  config.load_from_str (config_str);

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

  // configure the format based on the config
  if (!format)
    throw runtime_error ("format was not allocated");
  if (verbose)
    cerr << "spip::UDPReceiveDB::configure format->configure()" << endl;
  format->configure (config, "");

  // now write new params to config
  resolution = format->get_resolution();
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

  udp_stats = new UDPStats (format->get_header_size(), format->get_data_size());

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
        set_control_cmd(Record);
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
  uint64_t b_recv_total, b_recv_curr, b_recv_10sec;
  uint64_t b_drop_total, b_drop_curr, b_drop_10sec;
  uint64_t s_total, s_curr, s_10sec;
  double gb_recv_ps, gb_drop_ps, gb_drop;

  if (verbose)
    cerr << "spip::UDPReceiveDB::stats_thread starting polling" << endl;

  int report_time = 0;
  while (control_cmd != Quit)
  {
    b_recv_total = 0;
    b_drop_total = 0;
    s_total = 0;

#ifdef _DEBUG
    if (verbose > 1)
      cerr << "spip::UDPReceiveDB::stats_thread control_cmd != Quit" << endl;
#endif

    double gbits = 1e9;
    double gbits_per_10s = gbits * 10;
    report_time = 0;

    while (control_state == Recording)
    {
      if (report_time >= 10)
      {
        // get a snapshot of the data as quickly as possible
        b_recv_curr = udp_stats->get_data_transmitted();
        b_drop_curr = udp_stats->get_data_dropped();
        s_curr      = udp_stats->get_nsleeps();

        // calc the values for the last second
        b_recv_10sec = b_recv_curr - b_recv_total;
        b_drop_10sec = b_drop_curr - b_drop_total;
        s_10sec      = s_curr - s_total;

        // update the totals
        b_recv_total = b_recv_curr;
        b_drop_total = b_drop_curr;
        s_total      = s_curr;

        // calculate current rate
        gb_recv_ps = double(b_recv_10sec * 8) / gbits_per_10s;
        gb_drop_ps = double(b_drop_10sec * 8) / gbits_per_10s;
        gb_drop = double(b_drop_total * 8) / gbits;

        fprintf (stderr,"Recv %6.3lf [Gb/s] Dropped %6.3lf [Gb/s] %lf [Gb]\n", 
                 gb_recv_ps, gb_drop_ps, gb_drop);
        report_time = 0;
      }
      sleep (1);
      report_time++;
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

  if (verbose)
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
    cerr << "spip::UDPReceiveDB::close udp_stats->reset()" << endl;
  udp_stats->reset();

  if (verbose)
    cerr << "spip::UDPReceiveDB::close" << endl;
}

// receive UDP packets for the specified time at the specified data rate
bool spip::UDPReceiveDB::receive (int core)
{
  if (verbose)
    cerr << "spip::UDPReceiveDB::receive ()" << endl;

  // configure the cpu and memory binding
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

  // configure the UDP socket
  if (verbose)
    cerr << "spip::UDPReceiveDB::receive creating socket" << endl;
  // open socket within the context of this thread 
#ifdef HAVE_VMA
  sock = new UDPSocketReceiveVMA ();
#else
  sock = new UDPSocketReceive ();
#endif

  // handle multi or uni case
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
  
  // configure the socket buffer sizes
  size_t sock_bufsz = format->get_header_size() + format->get_data_size();
  if (verbose)
    cerr << "spip::UDPReceiveDB::receive sock->resize(" << sock_bufsz << ")" << endl;
  sock->resize (sock_bufsz);
  size_t kernel_bufsz = 64*1024*1024;
  if (verbose)
    cerr << "spip::UDPReceiveDB::receive sock->resize_kernel_buffer(" << kernel_bufsz << ")" << endl;
  sock->resize_kernel_buffer (kernel_bufsz);

  // block accounting
  curr_byte_offset = 0;
  next_byte_offset = 0;
  last_byte_offset = 0;

  control_state = Idle;

  spip::UDPSocketReceive::keep_receiving = true;

  // wait for a control command to begin
  pthread_mutex_lock (&mutex);
  while (control_cmd == None)
    pthread_cond_wait (&cond, &mutex);
  pthread_mutex_unlock (&mutex);

  curr_block = NULL;
  next_block = NULL;
  bytes_curr_block = 0;
  bytes_next_block = 0;
 
  // we have control command, so start the main loop
  while (spip::UDPSocketReceive::keep_receiving)
  {
    if (control_cmd != None)
      cerr << "spip::UDPReceiveDB::receive control_cmd=" << control_cmd << endl;
    // read thecontrol command
    switch (control_cmd)
    {
      case Record:
        if (verbose)
          cerr << "Start recording " << endl;
        if (control_state == Idle)
          next_block = (char *) db->open_block();
        control_state = Recording;
        set_control_cmd (None);

        break;

      case Stop:
        if (verbose)
          cerr << "Stop recording" << endl;
        // close the "next" block that would have been opened
        if (control_state == Recording)
        {
          db->close_block (data_block_bufsz);
          next_block = NULL;
        }
        control_state = Idle;
        spip::UDPSocketReceive::keep_receiving = false;
        set_control_cmd (None);
        break;

      case Quit:
        if (verbose)
          cerr << "Quiting" << endl;
        control_state = Quitting;
        spip::UDPSocketReceive::keep_receiving = false;
        set_control_cmd (None);
        break;

      case None:
        break;

      default:
        break;
    }

    if (control_state == Recording)
    {
      // rotate blocks
      curr_block = next_block;
      bytes_curr_block = bytes_next_block;

      // get a new next block
      next_block = (char *) db->open_block();
      bytes_next_block = 0;

      // receive into curr_block and next_block
      receive_block (format);

      // close curr_block
      db->close_block (data_block_bufsz);
      curr_block = NULL;
    }
    else
    {
      if (verbose)
        cerr << "spip::UDPReceiveDB::receive control_state not recording or monitoring" << endl;
      sleep (1);
    }
  }

#ifdef _DEBUG
  cerr << "spip::UDPReceiveDB::receive exiting" << endl;
#endif

  // close the socket
  sock->close_me();

  // close the data block
  if (verbose)
    cerr << "spip::UDPReceiveDB::receive this->close()"  << endl;
  close();

  if (control_state == Idle)
    return true;
  else
    return false;
}
     
// receive a block of UDP data
void spip::UDPReceiveDB::receive_block (UDPFormat * fmt)
{
  // increment the constraints for this block
  curr_byte_offset = next_byte_offset;
  next_byte_offset += data_block_bufsz;
  last_byte_offset += data_block_bufsz;

#ifdef _DEBUG
  cerr << "spip::UDPReceiveDB::receive_block filling buffer ["
       << curr_byte_offset << " - " << next_byte_offset
       << " - " << last_byte_offset << "] " << endl;
#endif

  unsigned bytes_received;
  bool filled_this_block = false;

  // tighter loop to fill this buffer
  while (!filled_this_block && sock->still_receiving())
  {
    // get a packet from the socket
    int got = sock->recv_from();
    if (got == 0)
    {
      set_control_cmd (Stop);
    }
    else
    {
      int64_t byte_offset = fmt->decode_packet (sock->buf_ptr, &bytes_received);

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
          bytes_curr_block += bytes_received;
          udp_stats->increment_bytes (bytes_received);
          fmt->insert_last_packet (curr_block + (byte_offset - curr_byte_offset));
          sock->consume_packet();
        }
        // packet belongs in next buffer
        else if ((byte_offset >= next_byte_offset) && (byte_offset < last_byte_offset))
        {
          bytes_next_block += bytes_received;
          udp_stats->increment_bytes (bytes_received);
          fmt->insert_last_packet (next_block + (byte_offset - next_byte_offset));
          sock->consume_packet();
        }
        // ignore packets the preceed this buffer
        else if (byte_offset < curr_byte_offset)
        {
          sock->consume_packet();
        }
        // packet is beyond next buffer
        else
        {
          filled_this_block = true;
        }
      }

      // close open data block buffer if is is now full
      if (bytes_curr_block >= int64_t(data_block_bufsz) || filled_this_block)
      {
#ifdef _DEBUG
        cerr << bytes_curr_block << " / " << data_block_bufsz << " => " <<
              ((float) bytes_curr_block / (float) data_block_bufsz) * 100 << endl;
        cerr << "spip::UDPReceiveDB::receive_block bytes_curr_block="
             << bytes_curr_block << " bytes_per_buf=" << data_block_bufsz
             << " bytes_next_block=" << bytes_next_block << endl;
#endif
        udp_stats->dropped_bytes (data_block_bufsz - bytes_curr_block);
        filled_this_block = true;
      }
    }
  }
}
