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
#include "spip/UDPReceiveDBStats.h"
#include "spip/Time.h"
#include "spip/Error.h"
#include "sys/time.h"

#include <unistd.h>
#include <cstring>
#include <iostream>
#include <stdexcept>
#include <new>
#include <pthread.h>
#include <sys/time.h>

//#define _DEBUG

using namespace std;

spip::UDPReceiveDBStats::UDPReceiveDBStats (const char * key_string) : UDPReceiveDB(key_string)
{
  block_format = NULL;
  monitor_curr_block = (char *) malloc (data_block_bufsz);
  monitor_next_block = (char *) malloc (data_block_bufsz);
}

spip::UDPReceiveDBStats::~UDPReceiveDBStats()
{
  if (block_format)
    delete block_format;
  block_format = NULL;

  if (monitor_curr_block)
    free (monitor_curr_block);
  monitor_curr_block = NULL;

  if (monitor_next_block)
    free (monitor_next_block);
  monitor_next_block = NULL;

  if (monitoring_udp_format)
    delete monitoring_udp_format;
  monitoring_udp_format = NULL;
}

int spip::UDPReceiveDBStats::configure (const char * config_str)
{
  spip::UDPReceiveDB::configure(config_str);

  // these parameters are requried for the statistics dumps 
  if (config.get ("NCHAN", "%u", &nchan) != 1)
    throw invalid_argument ("NCHAN did not exist in config");

  if (config.get ("NBIT", "%u", &nbit) != 1)
    throw invalid_argument ("NBIT did not exist in config");

  if (config.get ("NPOL", "%u", &npol) != 1)
    throw invalid_argument ("NPOL did not exist in config");

  if (config.get ("FREQ", "%lf", &freq) != 1)
    throw invalid_argument ("FREQ did not exist in config");

  if (config.get ("BW", "%lf", &bw) != 1)
    throw invalid_argument ("BW did not exist in config");

  if (config.get ("TSAMP", "%lf", &tsamp) != 1)
    throw invalid_argument ("TSAMP did not exist in config");

  // prepare the monitoring block format
  if (block_format)
  {
#ifdef _DEBUG
    cerr << "spip::UDPReceiveDBStats::configure block_format->set_resolution(" << resolution << ")" << endl;
#endif
    block_format->set_resolution (resolution);

    unsigned ntime = 256;
    unsigned nfreq = 256;
#ifdef _DEBUG
    cerr << "spip::UDPReceiveDBStats::configure block_format->prepare()" << endl;
#endif
    block_format->prepare (nbin, ntime, nfreq, freq, bw, tsamp);
  }

#ifdef _DEBUG
  cerr << "spip::UDPReceiveDBStats::configure monitoring_udp_format->configure()" << endl;
#endif
  monitoring_udp_format->configure (config, "");

  char command[128];
  stringstream ss;
  ss << stats_dir << "/" << freq;
  sprintf (command, "mkdir -p %s", ss.str().c_str());
  if (verbose)
    cerr << "spip::UDPReceiveDBStats::configure " << command << endl;
  int rval = system (command);
  if (rval != 0)
    throw Error (InvalidState, "spip::UDPReceiveDBStats::configure",
                 "could not create stats dir");
  return 0;
}

void spip::UDPReceiveDBStats::set_format (spip::UDPFormat * fmt, spip::UDPFormat * mon_fmt)
{
  if (format)
    delete format;
  format = fmt;

  if (monitoring_udp_format)
    delete monitoring_udp_format;
  monitoring_udp_format = mon_fmt;
}


void spip::UDPReceiveDBStats::set_block_format (spip::BlockFormat * fmt)
{
  if (verbose)
    cerr << "spip::UDPReceiveDBStats::set_block_format" << endl;
  if (block_format)
    delete block_format;
  block_format = fmt;
  nbin = block_format->get_nbin();
}

void spip::UDPReceiveDBStats::configure_stats_output (std::string dir, unsigned id)
{
  if (verbose)
    cerr << "spip::UDPReceiveDBStats::configure_stats_output" << endl;
  stats_dir = dir;
  stream_id = id;
}

void spip::UDPReceiveDBStats::control_thread()
{
  if (verbose)
    cerr << "spip::UDPReceiveDBStats::control_thread" << endl;
  if (control_port < 0)
  {
    cerr << "WARNING: no control port, using 32132" << endl;
    control_port = 32132;
  }

#ifdef _DEBUG
  cerr << "spip::UDPReceiveDBStats::control_thread creating TCPSocketServer" << endl;
#endif
  spip::TCPSocketServer * control_sock = new spip::TCPSocketServer();

  // open a listen sock on all interfaces for the control port
  if (verbose)
    cerr << "spip::UDPReceiveDBStats::control_thread open socket on port=" 
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

        //wait_control_state ();

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
  cerr << "spip::UDPReceiveDBStats::control_thread exiting" << endl;
#endif
}


bool spip::UDPReceiveDBStats::main (int core)
{
  // configure the cpu and memory binding
#ifdef HAVE_HWLOC
  spip::HardwareAffinity hw_affinity;
  if (core >= 0)
  {
    if (verbose)
      cerr << "spip::UDPReceiveDBStats::receive hw_affinity.bind_thread_to_cpu_core(" << core << ")" << endl;
    hw_affinity.bind_thread_to_cpu_core (core);
    hw_affinity.bind_to_memory (core);
  }
#endif

  if (verbose)
    cerr << "spip::UDPReceiveDBStats::main creating socket" << endl;
#ifdef HAVE_VMA
  sock = new UDPSocketReceiveVMA();
#else
  sock = new UDPSocketReceive();
#endif

  // always begin as Idle
  control_state = Idle;

  // wait for a control command to begin
  pthread_mutex_lock (&mutex);
  while (control_cmd == None)
    pthread_cond_wait (&cond, &mutex);
  pthread_mutex_unlock (&mutex);

  if (verbose)
    cerr << "spip::UDPReceiveDBStats::main received control command" << endl;

  bool keep_processing = true;
  while (keep_processing)
  {
    // read the control command
    ack_control_command ();

    if (verbose)
      cerr << "spip::UDPReceiveDBStats::main acknowledged control command" << endl;

    if (control_state == Monitoring)
      monitor();
    else if (control_state == Recording)
      receive ();
    else
      keep_processing = false;
  }
  
  if (verbose)
    cerr << "spip::UDPReceiveDBStats::main deleting socket" << endl;
  delete sock;

  if (control_state == Quitting)
    return false;
  else
    return true;

}

void spip::UDPReceiveDBStats::ack_control_command ()
{
  // read the control command
  switch (control_cmd)
  {
    case Monitor:
#ifdef _DEBUG
      cerr << "spip::UDPReceiveDBStats::receive control_cmd=Monitor"<< endl;
#endif
      cerr << "Start Monitoring" << endl;
      control_state = Monitoring;
      set_control_cmd (None);
      break;

    case Record:
#ifdef _DEBUG
      cerr << "spip::UDPReceiveDBStats::receive control_cmd=Record" << endl;
#endif
      if (control_state == Idle)
        next_block = (char *) db->open_block();
      cerr << "Start Recording" << endl;
      control_state = Recording;
      set_control_cmd (None);
      
      break;

    case Stop:
#ifdef _DEBUG
      cerr << "spip::UDPReceiveDBStats::receive control_cmd=Stop" << endl;
#endif
      // close the next block
      if (control_state == Recording)
      {
        db->close_block (data_block_bufsz);
        next_block = NULL;
      }
      cerr << "Stop Monitoring/Recording" << endl;
      control_state = Idle;
      spip::UDPSocketReceive::keep_receiving = false;
      set_control_cmd (None);
      break;

    case Quit:
#ifdef _DEBUG
      cerr << "spip::UDPReceiveDBStats::receive control_cmd=Quit" << endl;
#endif
      // close the next block
      if (control_state == Recording)
      {
        db->close_block (data_block_bufsz);
        next_block = NULL;
      }
      if (verbose)
        cerr << "Quiting" << endl;
      control_state = Quitting;
      spip::UDPSocketReceive::keep_receiving = false;
      set_control_cmd (None);
      break;

    case None:

    default:
      break;
  }
}

void spip::UDPReceiveDBStats::prepare_socket ()
{
  // configure the UDP socket
  if (data_mcast.size() > 0)
  {
    if (verbose)
      cerr << "spip::UDPReceiveDBStats::prepare_socket opening multicast socket" << endl;
    sock->open_multicast (data_host, data_mcast, data_port);
  }
  else
  {
    if (verbose)
      cerr << "spip::UDPReceiveDBStats::prepare_socket opening socket" << endl;
    sock->open (data_host, data_port);
  }

  // configure the socket buffer sizes
  size_t sock_bufsz = format->get_header_size() + format->get_data_size();
  if (verbose)
    cerr << "spip::UDPReceiveDBStats::prepare_socket sock->resize(" << sock_bufsz  + 64 << ")" << endl;
  sock->resize (sock_bufsz + 64);

  size_t kernel_bufsz = 64*1024*1024;
  if (verbose)
    cerr << "spip::UDPReceiveDBStats::prepare_socket sock->resize_kernel_buffer(" << kernel_bufsz << ")" << endl;
  sock->resize_kernel_buffer (kernel_bufsz);
}


// receive UDP packets for the specified time at the specified data rate
void spip::UDPReceiveDBStats::receive ()
{
  if (verbose)
    cerr << "spip::UDPReceiveDBStats::receive ()" << endl;

  // configure the UDP socket
  if (verbose)
    cerr << "spip::UDPReceiveDBStats::receive opening socket" << endl;
  prepare_socket();

  // block accounting
  curr_byte_offset = 0;
  next_byte_offset = 0;
  last_byte_offset = 0;

  // ensure that a receive_block call will start from the start
  udp_stats->reset();

  // ensure all packets buffered at the socket are cleared
  sock->clear_buffered_packets();

  spip::UDPSocketReceive::keep_receiving = true;
  while (spip::UDPSocketReceive::keep_receiving)
  {
    // rotate blocks
    curr_block = next_block;
    bytes_curr_block = bytes_next_block;

    // get a new next block
    next_block = (char *) db->open_block();
    bytes_next_block = 0;

    receive_block (format);

    // close curr_block
    db->close_block (data_block_bufsz);
    curr_block = NULL;

    // check or a control cmd, to handle a change in state
    ack_control_command ();
  }

#ifdef _DEBUG
  cerr << "spip::UDPReceiveDBStats::receive exiting" << endl;
#endif

  // close the socket
  sock->close_me();

  // close the data block
  if (verbose)
    cerr << "spip::UDPReceiveDBStats::receive this->close()"  << endl;
  close();

  if (verbose)
    cerr << "spip::UDPReceiveDBStats::receive returning"  << endl;
}

// receive UDP packets for the specified time at the specified data rate
void spip::UDPReceiveDBStats::monitor ()
{
  if (verbose)
    cerr << "spip::UDPReceiveDBStats::monitor ()" << endl;

  unsigned monitor_period = 9;  // seconds
  unsigned monitor_wait = monitor_period + 1; // seconds

  bool keep_monitoring = true;
  spip::UDPSocketReceive::keep_receiving = true;

  // record data to the monitoring block
  curr_block = monitor_curr_block;
  next_block = monitor_next_block;

  while (keep_monitoring && spip::UDPSocketReceive::keep_receiving)
  {
    // if time has come to update the monitoring data
    if (monitor_wait > monitor_period)
    {
      // open the UDP socket
      if (verbose)
        cerr << "spip::UDPReceiveDBStats::monitor creating socket" << endl;
      prepare_socket();

      // block accounting
      curr_byte_offset = 0;
      next_byte_offset = 0;
      last_byte_offset = 0;

      // we have control command, so start the main loop
      // form a temporary header based on the configuration
      AsciiHeader monitor_header (config);

      // generate a UTC_START in 2s from now
      spip::Time utc_start (time(0));
      std::string utc_str = utc_start.get_gmtime();
      utc_start.add_seconds (2);
      utc_str = utc_start.get_gmtime();

      // set this UTC_START in the monitoring header
      if (monitor_header.set ("UTC_START", "%s", utc_str.c_str()) < 0)
      {
        throw invalid_argument ("failed to write UTC_START to monitor_header");
      }

      if (verbose)
       cerr << "spip::UDPReceiveDBStats::monitor capturing monitoring block at UTC=" << utc_str.c_str() << endl;

#ifdef _DEBUG
      cerr << "spip::UDPReceiveDBStats::monitor monitoring_udp_format->prepare(monitor_header, '')" << endl;
#endif

      // prepare the monitoring format with the monitoring header
      monitoring_udp_format->prepare (monitor_header, "");

      // reset the UDP statistics counter
      udp_stats->reset();

#ifdef _DEBUG
      cerr << "spip::UDPReceiveDBStats::monitor receive_block()" << endl;
#endif

      // receive 1 data_block's worth of samples
      receive_block (monitoring_udp_format);

      // immediately close the UDP socket
      sock->close_me();

      uint64_t b_recv_curr = udp_stats->get_data_transmitted();
      uint64_t b_drop_curr = udp_stats->get_data_dropped();
      if (b_drop_curr > 0)
        cerr << "spip::UDPReceiveDBStats::monitor recv=" << b_recv_curr << " drop=" << b_drop_curr << " buf=" << data_block_bufsz << endl;

#ifdef _DEBUG
      cerr << "spip::UDPReceiveDBStats::monitor analyze_block()" << endl;
#endif

      // process the data block, writing files to disk
      analyze_block();
    
      // reset the monitoring wait counter to zero
      monitor_wait = 0;
    }
    else
    {
      sleep (1);
      monitor_wait++;
    }

    // check or a control cmd, to handle a change in state
    ack_control_command ();

    // return if monitoring is over
    if (control_state != Monitoring)
      keep_monitoring = false;
  }

  curr_block = NULL;
  next_block = NULL;
}

void spip::UDPReceiveDBStats::analyze_block ()
{
  if (verbose)
    cerr << "spip::UDPReceiveDBStats::analyze_block processing " 
         << data_block_bufsz << " bytes" << endl;

  if (verbose > 1)
    cerr << "spip::UDPReceiveDBStats::analyze_block reset block_format" << endl;
  block_format->reset();

  if (verbose > 1)
    cerr << "spip::UDPReceiveDBStats::analyze_block block_format->unpack_hgft()" << endl;
  block_format->unpack_hgft (monitor_curr_block, data_block_bufsz);

  if (verbose > 1)
    cerr << "spip::UDPReceiveDBStats::analyze_block block_format->unpack_ms()" << endl;
  block_format->unpack_ms (monitor_curr_block, data_block_bufsz);

  // write the data files to disk 
  time_t now = time(0);
  char local_time[32];
  strftime (local_time, 32, DADA_TIMESTR, localtime(&now));

  if (verbose > 1)
    cerr << "spip::UDPReceiveDBStats::analyze_block generating output" << endl;

  stringstream ss;

  ss << stats_dir << "/" << freq << "/" << local_time << "." << stream_id << ".??.stats";
  if (verbose)
    cerr << "spip::UDPReceiveDBStats::analyze_block generating stats files " << ss.str() << endl;

  // generate the histogram files
  ss.str("");
  ss << stats_dir << "/" << freq << "/" << local_time << "." << stream_id << ".hg.stats";
  if (verbose > 1)
    cerr << "spip::UDPReceiveDBStats::analyze_block creating HG stats file " << ss.str() << endl;
  block_format->write_histograms (ss.str());

  // generate the freq/time files
  //ss.str("");
  //ss << stats_dir << "/" << freq << "/" << local_time << "." << stream_id << ".ft.stats";
  //if (verbose > 1)
  //  cerr << "spip::UDPReceiveDBStats::analyze_block creating FT stats file " << ss.str() << endl;
  //block_format->write_freq_times (ss.str());

  // generate the bandpass files
  ss.str("");
  ss << stats_dir << "/" << freq << "/" << local_time << "." << stream_id << ".bp.stats";
  if (verbose > 1)
    cerr << "spip::UDPReceiveDBStats::analyze_block creating BP stats file " << ss.str() << endl;
  block_format->write_bandpasses (ss.str());


  // generate the time series files
  ss.str("");
  ss << stats_dir << "/" << freq << "/" << local_time << "." << stream_id << ".ts.stats";
  if (verbose > 1)
    cerr << "spip::UDPReceiveDBStats::analyze_block creating TS stats file " << ss.str() << endl;
  block_format->write_time_series (ss.str());

  // generate the mean and stddev files
  ss.str("");
  ss << stats_dir << "/" << freq << "/" << local_time << "." << stream_id << ".ms.stats";
  if (verbose > 1)
    cerr << "spip::UDPReceiveDBStats::analyze_block creating MS stats file " << ss.str() << endl;
  block_format->write_mean_stddevs (ss.str());
}

