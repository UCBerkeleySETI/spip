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

#include "spip/TCPSocketServer.h"
#include "spip/AsciiHeader.h"
#include "spip/IBVReceiveDB.h"
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

spip::IBVReceiveDB::IBVReceiveDB(const char * key_string, 
                                 boost::asio::io_service& io_service) 
                                 : queue(io_service)
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
  verbose = verbose;
}

spip::IBVReceiveDB::~IBVReceiveDB()
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

int spip::IBVReceiveDB::configure (const char * config_str)
{
  // save the header for use on the first open block
  if (verbose)
    cerr << "spip::IBVReceiveDB::configure config.load_from_str(config_str)" << endl;
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
    cerr << "spip::IBVReceiveDB::configure format->configure()" << endl;
  format->configure (config, "");

  // now write new params to config
  resolution = format->get_resolution();
  if (config.set("RESOLUTION", "%lu", resolution) < 0)
    throw invalid_argument ("failed to write RESOLUTION to config");
  if (verbose)
    cerr << "spip::IBVReceiveDB::configure resolution=" << resolution << endl;

  if (db->get_data_bufsz() % resolution != 0)
  {
    cerr << "db->get_data_bufsz=" << db->get_data_bufsz () << endl;
    cerr << "RESOLUTION=" << resolution << endl;
    throw invalid_argument ("Data block buffer size must be multiple of RESOLUTION");
  }

  udp_stats = new UDPStats (format->get_header_size(), format->get_data_size());

  // configure the queue
  // TODO make this configurable
  size_t num_packets = 8192;
  packet_size = format->get_packet_size();
  header_size = format->get_header_size();
  queue.configure (num_packets, packet_size, header_size);

  // open the IBV Queue 
  if (data_mcast.size() > 0)
  {
    if (verbose)
      cerr << "spip::IBVReceiveDB::configure queue.open_multicast (" 
           << data_host << ", " << data_mcast << ", " << data_port << ")" 
           << endl;
    queue.open_multicast (data_host, data_mcast, data_port);
  }
  else
  {
    if (verbose)
      cerr << "spip::IBVReceiveDB::configure queue.open(" << data_host << ", " 
           << data_port << ")" << endl;
    queue.open (data_host, data_port);
  }

  // allocate requirement memory resources
  queue.allocate ();

  if (data_mcast.size() > 0)
    queue.join_multicast (data_host, data_port);

  return 0;
}

void spip::IBVReceiveDB::set_format (spip::UDPFormat * fmt)
{
  if (format)
    delete format;
  format = fmt;
}

void spip::IBVReceiveDB::start_control_thread (int port)
{
  control_port = port;
  pthread_create (&control_thread_id, 0, control_thread_wrapper, this);
}

void spip::IBVReceiveDB::stop_control_thread ()
{
  set_control_cmd (Quit);
}

void spip::IBVReceiveDB::set_control_cmd (spip::ControlCmd cmd)
{
  pthread_mutex_lock (&mutex);
  control_cmd = cmd;
  pthread_cond_signal (&cond);
  pthread_mutex_unlock (&mutex);
}

// start a control thread that will receive commands from the TCS/LMC
void spip::IBVReceiveDB::control_thread()
{
  if (control_port < 0)
  {
    cerr << "WARNING: no control port, using 32132" << endl;
    control_port = 32132;
  }

#ifdef _DEBUG
  cerr << "spip::IBVReceiveDB::control_thread creating TCPSocketServer" << endl;
#endif
  spip::TCPSocketServer * control_sock = new spip::TCPSocketServer();

  // open a listen sock on all interfaces for the control port
  if (verbose)
    cerr << "spip::IBVReceiveDB::control_thread open socket on port=" 
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
  cerr << "spip::IBVReceiveDB::control_thread exiting" << endl;
#endif
}


void spip::IBVReceiveDB::start_stats_thread ()
{
  pthread_create (&stats_thread_id, NULL, stats_thread_wrapper, this);
}

void spip::IBVReceiveDB::stop_stats_thread ()
{
  set_control_cmd(Quit);
  void * result;
  pthread_join (stats_thread_id, &result);
}

/* 
 *  Thread to print simple capture statistics
 */
void spip::IBVReceiveDB::stats_thread()
{
  uint64_t b_recv_total, b_recv_curr, b_recv_base;
  uint64_t p_recv_total, p_recv_curr, p_recv_base;
  uint64_t b_drop_total, b_drop_curr, b_drop_base;
  uint64_t p_drop_total, p_drop_curr, p_drop_base;
  uint64_t s_total, s_curr, s_base;
  double gb_recv_ps, gb_drop_ps, gb_drop;
  double kp_recv_ps, kp_drop_ps, kp_drop;
  double s_ps;

  if (verbose)
    cerr << "spip::IBVReceiveDB::stats_thread starting polling" << endl;

  int report_time = 0;
  int report_base = 1;
  while (control_cmd != Quit)
  {
    b_recv_total = p_recv_total = 0;
    b_drop_total = p_drop_total = 0;
    s_total = 0;

#ifdef _DEBUG
    if (verbose > 1)
      cerr << "spip::IBVReceiveDB::stats_thread control_cmd != Quit" << endl;
#endif

    double gbit = 1e9;
    double kpkt = 1e3;
    double gbits_per_base = gbit * report_base;
    double kpkts_per_base = kpkt * report_base;
    double sleeps_per_base = double(report_base);

    report_time = 0;

    while (control_state == Recording)
    {
      if (report_time % report_base == 0)
      {
        // get a snapshot of the data as quickly as possible
        b_recv_curr = udp_stats->get_data_transmitted();
        p_recv_curr = udp_stats->get_packets_transmitted();
        b_drop_curr = udp_stats->get_data_dropped();
        p_drop_curr = udp_stats->get_packets_dropped();
        s_curr      = udp_stats->get_nsleeps();

        // calc the values for the last base period
        b_recv_base = b_recv_curr - b_recv_total;
        p_recv_base = p_recv_curr - p_recv_total;
        b_drop_base = b_drop_curr - b_drop_total;
        p_drop_base = p_drop_curr - p_drop_total;
        s_base      = s_curr - s_total;

        // update the totals
        b_recv_total = b_recv_curr;
        p_recv_total = p_recv_curr;
        b_drop_total = b_drop_curr;
        p_drop_total = p_drop_curr;
        s_total      = s_curr;

        // calculate current rate
        gb_recv_ps = double(b_recv_base * 8) / gbits_per_base;
        gb_drop_ps = double(b_drop_base * 8) / gbits_per_base;
        gb_drop = double(b_drop_total * 8) / gbit;

        kp_recv_ps = double(p_recv_base) / kpkts_per_base;
        kp_drop_ps = double(p_drop_base) / kpkts_per_base;
        kp_drop = double(p_drop_total) / kpkt;

        s_ps = double(s_base) / sleeps_per_base;

        if (report_time % (20 * report_base) == 0)
        {
          fprintf (stderr, "Received\tDropped \t\tSleeps\n");
          fprintf (stderr, "Gb/s   (kp/s)\tGb/s   (kp/s)       Gb\n");
        }
        fprintf (stderr, "%6.3lf (%6.2lf)\t%6.3lf (%6.2lf) %6.2lf\t%6.0lf\n", 
                 gb_recv_ps, kp_recv_ps, gb_drop_ps, kp_drop_ps, gb_drop, s_ps);
      }
      sleep (1);
      report_time++;
    }
    sleep(1);
  }
}

bool spip::IBVReceiveDB::open ()
{
  if (verbose > 1)
    cerr << "spip::IBVReceiveDB::open()" << endl;
  
  if (control_cmd == Stop)
  {
    return false;
  }

  if (header.get_header_length() == 0)
  {
    if (verbose)
      cerr << "spip::IBVReceiveDB::open header.clone(config)" << endl;
    header.clone(config);
  }

  // check if UTC_START has been set
  char * buffer = (char *) malloc (128);
  if (header.get ("UTC_START", "%s", buffer) == -1)
  {
    cerr << "spip::IBVReceiveDB::open no UTC_START in header" << endl;
    time_t now = time(0);
    spip::Time utc_start (now);
    utc_start.add_seconds (10);
    std::string utc_str = utc_start.get_gmtime();
    cerr << "Generated UTC_START=" << utc_str  << endl;
    if (header.set ("UTC_START", "%s", utc_str.c_str()) < 0)
      throw invalid_argument ("failed to write UTC_START to header");
  }
  else
    cerr << "spip::IBVReceiveDB::open UTC_START=" << buffer << endl;

  uint64_t obs_offset;
  if (header.get("OBS_OFFSET", "%lu", &obs_offset) == -1)
  {
    obs_offset = 0;
    if (header.set ("OBS_OFFSET", "%lu", obs_offset) < 0)
      throw invalid_argument ("failed to write OBS_OFFSET=0 to header");
  }

  if (header.get ("SOURCE", "%s", buffer) == -1)
  {
    cerr << "spip::IBVReceiveDB::open no SOURCE in header, using J047-4715" << endl;
    if (header.set ("SOURCE", "%s", "J0437-4715") < 0)
      throw invalid_argument ("failed to write SOURCE to header");
  }

  if (verbose)
    cerr << "spip::IBVReceiveDB::open format->prepare(header, )" << endl;
  format->prepare(header, "");

  open (header.raw());
  free (buffer);

  return true;
}

// write the ascii header to the datablock, then
void spip::IBVReceiveDB::open (const char * header_str)
{
  // open the data block for writing  
  if (verbose)
    cerr << "spip::IBVReceiveDB::open db->open()" << endl;
  db->open();

  // write the header
  if (verbose)
    cerr << "spip::IBVReceiveDB::open db->write_header()" << endl;
  db->write_header (header_str);
}

void spip::IBVReceiveDB::close ()
{
  if (verbose)
    cerr << "spip::IBVReceiveDB::close()" << endl;

  if (db->is_block_open())
  {
    db->close_block(db->get_data_bufsz());
  }

  // close the data block, ending the observation
  if (verbose)
    cerr << "spip::IBVReceiveDB::close db->close" << endl;
  db->close();

  // clear the header
  if (verbose)
    cerr << "spip::IBVReceiveDB::close header.reset()" << endl;
  header.reset();

  // reset the UDP stats
  if (verbose)
    cerr << "spip::IBVReceiveDB::close udp_stats->reset()" << endl;
  udp_stats->reset();

  if (verbose)
    cerr << "spip::IBVReceiveDB::close" << endl;
}

// receive UDP packets for the specified time at the specified data rate
bool spip::IBVReceiveDB::receive (int core)
{
  if (verbose)
    cerr << "spip::IBVReceiveDB::receive ()" << endl;

  // configure the cpu and memory binding
#ifdef HAVE_HWLOC
  spip::HardwareAffinity hw_affinity;
  if (core >= 0)
  {
    if (verbose)
      cerr << "spip::IBVReceiveDB::receive hw_affinity.bind_thread_to_cpu_core(" << core << ")" << endl;
    hw_affinity.bind_thread_to_cpu_core (core);
    hw_affinity.bind_to_memory (core);
  }
#endif

  // configure the overflow buffer
  overflow = new UDPOverflow();
  uint64_t overflow_bufsz = format->get_resolution() * 4;
  while (overflow_bufsz < 8 * 1024 * 1024)
    overflow_bufsz *= 2;
  overflow->resize (overflow_bufsz);
  overflow_block = (char *) overflow->get_buffer();

  // block accounting
  curr_byte_offset = 0;
  next_byte_offset = 0;
  overflow_maxbyte = 0;

  control_state = Idle;

  spip::IBVQueue::keep_receiving = true;

  // wait for a control command to begin
  pthread_mutex_lock (&mutex);
  while (control_cmd == None)
    pthread_cond_wait (&cond, &mutex);
  pthread_mutex_unlock (&mutex);
 
  // we have control command, so start the main loop
  while (spip::IBVQueue::keep_receiving)
  {
#ifdef _DEBUG
    if (control_cmd != None)
      cerr << "spip::IBVReceiveDB::receive control_cmd=" << control_cmd << endl;
#endif

    // read the control command
    switch (control_cmd)
    {
      case Record:
        if (verbose)
          cerr << "Start recording " << endl;
        control_state = Recording;
        set_control_cmd (None);
        break;

      case Stop:
        if (verbose)
          cerr << "Stop recording" << endl;
        control_state = Idle;
        spip::IBVQueue::keep_receiving = false;
        set_control_cmd (None);
        break;

      case Quit:
        if (verbose)
          cerr << "Quiting" << endl;
        control_state = Quitting;
        spip::IBVQueue::keep_receiving = false;
        set_control_cmd (None);
        break;

      case None:
        break;

      default:
        break;
    }

    if (control_state == Recording)
    {
      block = (char *) db->open_block();
      receive_block (format);
      db->close_block (data_block_bufsz);
    }
    else
    {
      if (verbose)
        cerr << "spip::IBVReceiveDB::receive control_state not recording or monitoring" << endl;
      sleep (1);
    }
  }

#ifdef _DEBUG
  cerr << "spip::IBVReceiveDB::receive exiting" << endl;
#endif

  // close the data block
  if (verbose)
    cerr << "spip::IBVReceiveDB::receive this->close()"  << endl;
  close();

  if (control_state == Idle)
    return true;
  else
    return false;
}
     
// receive a block of UDP data
void spip::IBVReceiveDB::receive_block (UDPFormat * fmt)
{
  // copy any data from the overflow into the new block
  int64_t bytes_this_buf = overflow->copy_to (block);
  udp_stats->increment_bytes (bytes_this_buf);

  // increment the constraints for this block
  curr_byte_offset = next_byte_offset;
  next_byte_offset += data_block_bufsz;
  overflow_maxbyte = next_byte_offset + overflow->get_bufsz();

#ifdef _DEBUG
  cerr << "spip::IBVReceiveDB::receive_block filling buffer ["
       << curr_byte_offset << " - " << next_byte_offset
       << " - " << overflow_maxbyte << "] "
       << " overflow_bufsz=" << overflow->get_bufsz() << endl;
#endif

  unsigned bytes_received;
  bool filled_this_block = false;

  // tighter loop to fill this buffer
  while (!filled_this_block && spip::IBVQueue::keep_receiving)
  {
    // get a packet from the socket
    int got = queue.open_packet();
    if (got > 0)
    {
      int64_t byte_offset = fmt->decode_packet ((char *) queue.buf_ptr, &bytes_received);

      // ignore if byte_offset is -ve
      if (byte_offset < 0)
      {
        queue.close_packet();

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
          udp_stats->increment_bytes (bytes_received);
          fmt->insert_last_packet (block + (byte_offset - curr_byte_offset));
          queue.close_packet();
        }
        // packet belongs in overflow buffer
        else if ((byte_offset >= next_byte_offset) && (byte_offset < overflow_maxbyte))
        {
          fmt->insert_last_packet (overflow_block + (byte_offset - next_byte_offset));
          overflow->copied_from (byte_offset - next_byte_offset, bytes_received);
          queue.close_packet();
        }
        // ignore packets the preceed this buffer
        else if (byte_offset < curr_byte_offset)
        {
          queue.close_packet();
        }
        // packet is beyond overflow buffer
        else
        {
          filled_this_block = true;
        }
      }

      // close open data block buffer if is is now full
      if (bytes_this_buf >= int64_t(data_block_bufsz) || filled_this_block)
      {
#ifdef _DEBUG
        cerr << bytes_this_buf << " / " << data_block_bufsz << " => " <<
              ((float) bytes_this_buf / (float) data_block_bufsz) * 100 << endl;
        cerr << "spip::IBVReceiveDB::receive_block bytes_this_buf="
             << bytes_this_buf << " bytes_per_buf=" << data_block_bufsz
             << " overflown=" << overflow->get_last_byte() << endl;
#endif
        udp_stats->dropped_bytes (data_block_bufsz - bytes_this_buf);
        udp_stats->sleeps(queue.process_sleeps());
        filled_this_block = true;
      }
    }
    else if (got == 0)
    {
      //cerr << "spip::IBVReceiveDB::receive no packets in queue" << endl;
    }
    else
    {
      set_control_cmd (Stop);
    }
  }
}
