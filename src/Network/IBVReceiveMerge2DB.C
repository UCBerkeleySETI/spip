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

#include "spip/IBVQueue.h"
#include "spip/UDPSocketSend.h"

#include "spip/TCPSocketServer.h"
#include "spip/IBVReceiveMerge2DB.h"
#include "spip/Time.h"

#include <pthread.h>
#include <cstring>
#include <iostream>
#include <stdexcept>
#include <new>

//#define _DEBUG

using namespace std;

spip::IBVReceiveMerge2DB::IBVReceiveMerge2DB (const char * key1, const char * key2)
{
  db1 = new DataBlockWrite (key1);
  db2 = new DataBlockWrite (key2);

  db1->connect();
  db1->lock();
  db1->page();

  db2->connect();
  db2->lock();
  db2->page();

  //db1->set_device(2);
  //db2->set_device(3);

  control_port = -1;

  control_cmd = None;
  control_state = Idle;
  boost::asio::io_service io;

  for (unsigned i=0; i<2; i++)
  {
    control_states[i] = Idle;
    queues[i] = new IBVQueue(io);

    pthread_cond_init( &(cond_recvs[i]), NULL);
    pthread_mutex_init( &(mutex_recvs[i]), NULL);

    data_mcasts[i] = string ();
    formats[i] = NULL;
    stats[i] = NULL;
  }

  pthread_cond_init( &cond_db, NULL);
  pthread_mutex_init( &mutex_db, NULL);

  chunk_size = 0;

  verbose = 0;
}

spip::IBVReceiveMerge2DB::~IBVReceiveMerge2DB()
{
#ifdef _DEBUG
  cerr << "spip::IBVReceiveMerge2DB::~IBVReceiveMerge2DB" << endl;
#endif

  for (unsigned i=0; i<2; i++)
  {
    delete stats[i];
    delete formats[i];
  }

  db1->unlock();
  db1->disconnect();

  db2->unlock();
  db2->disconnect();

  delete db1;
  delete db2;
}

int spip::IBVReceiveMerge2DB::configure (const char * config_str)
{
  // save the config for use on the first open block
  config.load_from_str (config_str);

  if (verbose)
    cerr << "spip::IBVReceiveMerge2DB::configure" << endl;

  if (config.get ("NCHAN", "%u", &nchan) != 1)
    throw invalid_argument ("NCHAN did not exist in config");

  if (config.get ("NBIT", "%u", &nbit) != 1)
    throw invalid_argument ("NBIT did not exist in config");

  if (config.get ("NPOL", "%u", &npol) != 1)
    throw invalid_argument ("NPOL did not exist in config");

  if (config.get ("NDIM", "%u", &ndim) != 1)
    throw invalid_argument ("NDIM did not exist in config");

  if (config.get ("TSAMP", "%lf", &tsamp) != 1)
    throw invalid_argument ("TSAMP did not exist in config");

  if (config.get ("BW", "%lf", &bw) != 1)
    throw invalid_argument ("BW did not exist in config");

  if (config.get ("FREQ", "%lf", &freq) != 1)
    throw invalid_argument ("FREQ did not exist in config");
  if (config.get ("START_CHANNEL", "%u", &start_channel) != 1)
    throw invalid_argument ("START_CHANNEL did not exist in config");
  if (config.get ("END_CHANNEL", "%u", &end_channel) != 1)
    throw invalid_argument ("END_CHANNEL did not exist in config");

  char * buffer = (char *) malloc (128);

  if (config.get ("DATA_HOST_0", "%s", buffer) != 1)
    throw invalid_argument ("DATA_HOST_0 did not exist in config");
  data_hosts[0] = string (buffer);
  if (config.get ("DATA_HOST_1", "%s", buffer) != 1)
    throw invalid_argument ("DATA_HOST_1 did not exist in config");
  data_hosts[1] = string (buffer);

  if (config.get ("DATA_PORT_0", "%d", &data_ports[0]) != 1)
    throw invalid_argument ("DATA_PORT_0 did not exist in config");
  if (config.get ("DATA_PORT_1", "%d", &data_ports[1]) != 1)
    throw invalid_argument ("DATA_PORT_1 did not exist in config");

  if (config.get ("DATA_MCAST_0", "%s", buffer) == 1)
    data_mcasts[0] = string (buffer);
  if (config.get ("DATA_MCAST_1", "%s", buffer) == 1)
    data_mcasts[1] = string (buffer);

  bits_per_second  = (unsigned) ((nchan * npol * ndim * nbit * 1000000) / tsamp);
  bytes_per_second = bits_per_second / 8;

  int header_npol = 1;
  config.set("NPOL", "%d", header_npol);

  formats[0]->configure(config, "_0");
  formats[1]->configure(config, "_1");

  npol = 2;
  if (config.set("NPOL", "%u", npol) < 0)
    throw invalid_argument ("failed to write NPOL to config"); 

  // packet sizing for buffering
  size_t packet_size = formats[0]->get_packet_size();
  size_t header_size = formats[0]->get_header_size();
  size_t num_packets = 128;

  for (unsigned i=0; i<2; i++)
  {
    stats[i] = new UDPStats (formats[i]->get_header_size(), formats[i]->get_data_size());
    queues[i]->configure (num_packets, packet_size, header_size);

    // open the IBV Queue 
    if (data_mcasts[i].size() > 0)
    {
      cerr << "spip::IBVReceiver::configure queue->open_multicast (" 
           << data_hosts[i] << ", " << data_mcasts[i] << ", " << data_ports[i]
           << ")" << endl;
      queues[i]->open_multicast (data_hosts[i], data_mcasts[i], data_ports[i]);
    }
    else
    {
      if (verbose)
        cerr << "spip::IBVReceiver::configure queue->open(" << data_hosts[i] 
             << ", " << data_ports[i] << ")" << endl;
      queues[i]->open (data_hosts[i], data_ports[i]);
    }

    // allocate requirement memory resources
    if (verbose)
      cerr << "spip::IBVReceiver::configure queue->allocate()" << endl;
    queues[i]->allocate ();
  }

  return 0;
}

void spip::IBVReceiveMerge2DB::set_formats (spip::UDPFormat * fmt1, spip::UDPFormat * fmt2)
{
  if (formats[0])
    delete formats[0];
  formats[0] = fmt1;

  if (formats[1])
    delete formats[1];
  formats[1] = fmt2;
}

void spip::IBVReceiveMerge2DB::start_control_thread (int port)
{
  control_port = port;
  pthread_create (&control_thread_id, NULL, control_thread_wrapper, this);
}

void spip::IBVReceiveMerge2DB::stop_control_thread ()
{
  control_cmd = Quit;
  void * result;
  pthread_join (control_thread_id, &result);
}

void spip::IBVReceiveMerge2DB::set_control_cmd (spip::ControlCmd cmd)
{
  pthread_mutex_lock (&mutex_db);
  control_cmd = cmd;
  pthread_cond_signal (&cond_db);
  pthread_mutex_unlock (&mutex_db);

  if ((cmd == Stop) || (cmd == Quit))
  {
    spip::IBVQueue::keep_receiving = false;
    send_terminal_packets();
  }
}

// start a control thread that will receive commands from the TCS/LMC
void spip::IBVReceiveMerge2DB::control_thread()
{
#ifdef _DEBUG
  cerr << "spip::IBVReceiveMerge2DB::control_thread starting" << endl;
#endif

  if (control_port < 0)
  {
    cerr << "ERROR: no control port specified" << endl;
    return;
  }

#ifdef _DEBUG
  cerr << "spip::IBVReceiveMerge2DB::control_thread creating TCPSocketServer" << endl;
#endif
  spip::TCPSocketServer * control_sock = new spip::TCPSocketServer();

  // open a listen sock on all interfaces for the control port
  if (verbose)
    cerr << "opened control socket on port=" << control_port << endl;
  control_sock->open ("any", control_port, 1);

  int fd = -1;
  int verbose = 1;

  char * cmd  = (char *) malloc (32);

  // wait for a connection
  while (control_cmd != Quit && fd < 0)
  {
    // accept with a 1 second timeout
#ifdef _DEBUG
    cerr << "control_thread : ctrl_sock->accept_client(1)" << endl;
#endif
    fd = control_sock->accept_client (1);
#ifdef _DEBUG
    cerr << "control_thread : fd=" << fd << endl;
#endif
    if (fd >= 0 )
    {
      if (verbose > 1)
        cerr << "control_thread : reading data from socket" << endl;
      string received = control_sock->read_client (DADA_DEFAULT_HEADER_SIZE);
      const char * cmds = received.c_str();
      if (verbose)
        cerr << "control_thread: bytes_read=" << strlen(cmds) << endl;
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
        header1.clone(config);
        header2.clone(config);
        header1.append_from_str (cmds);
        header2.append_from_str (cmds);
        if (header1.del ("COMMAND") < 0)
          throw runtime_error ("Could not remove COMMAND from header1");
        if (header2.del ("COMMAND") < 0)
          throw runtime_error ("Could not remove COMMAND from header2");

        if (verbose)
          cerr << "control_thread: open()" << endl;
        open ();

        // write header
        if (verbose)
          cerr << "control_thread: control_cmd = Start" << endl;
        set_control_cmd (Start);
      }
      else if (strcmp (cmd, "STOP") == 0)
      {
        if (verbose)
          cerr << "control_thread: control_cmd = Stop" << endl;
        set_control_cmd (Stop);
      }
      else if (strcmp (cmd, "QUIT") == 0)
      {
        if (verbose)
          cerr << "control_thread: control_cmd = Quit" << endl;
        set_control_cmd (Quit);
      }
    }
  }
#ifdef _DEBUG
  cerr << "spip::IBVReceiveMerge2DB::control_thread exiting" << endl;
#endif
}

bool spip::IBVReceiveMerge2DB::open ()
{
  if (control_cmd == Stop)
  {
    return false;
  } 

  if (header1.get_header_length() == 0)
    header1.clone(config);
  if (header2.get_header_length() == 0)
    header2.clone(config);
 
  // check if UTC_START has been set
  char * buffer = (char *) malloc (128);
  if (header1.get ("UTC_START", "%s", buffer) == -1)
  {
    time_t now = time(0);
    spip::Time utc_start (now);
    utc_start.add_seconds (5);
    std::string utc_str = utc_start.get_gmtime();
    cerr << "Generated UTC_START=" << utc_str  << endl;
    if (header1.set ("UTC_START", "%s", utc_str.c_str()) < 0)
      throw invalid_argument ("failed to write UTC_START to header1");
    if (header2.set ("UTC_START", "%s", utc_str.c_str()) < 0)
      throw invalid_argument ("failed to write UTC_START to header2");
  }
  else
    cerr << "UTC_START=" << buffer << endl; 

  uint64_t obs_offset;
  if (header1.get("OBS_OFFSET", "%lu", &obs_offset) == -1)
  {
    obs_offset = 0;
    if (header1.set ("OBS_OFFSET", "%lu", obs_offset) < 0)
      throw invalid_argument ("failed to write OBS_OFFSET=0 to header1");
  }
  if (header2.get("OBS_OFFSET", "%lu", &obs_offset) == -1)
  {
    obs_offset = 0;
    if (header2.set ("OBS_OFFSET", "%lu", obs_offset) < 0)
      throw invalid_argument ("failed to write OBS_OFFSET=0 to header2");
  }

  if (header1.get ("SOURCE", "%s", buffer) == -1)
  {
    cerr << "No SOURCE in header1, using J047-4715" << endl;
    if (header1.set ("SOURCE", "%s", "J0437-4715") < 0)
      throw invalid_argument ("failed to write SOURCE to header1");
  }
  if (header2.get ("SOURCE", "%s", buffer) == -1)
  {
    cerr << "No SOURCE in header2, using J047-4715" << endl;
    if (header2.set ("SOURCE", "%s", "J0437-4715") < 0)
      throw invalid_argument ("failed to write SOURCE to header2");
  }

#ifdef _DEBUG
  cerr << "spip::IBVReceiveMerge2DB::open formats->prepare()" << endl;
#endif

  formats[0]->prepare (header1, "_0");
  formats[1]->prepare (header2, "_1");

  // constants for this implementation
  unsigned nsubband = 2;
  unsigned new_npol = 2;
  unsigned new_nchan = nchan / nsubband;
  double new_bw = bw / nsubband;
  double new_freqs[2];
  unsigned new_start_channels[2];
  unsigned new_end_channels[2];

  new_freqs[0] = freq - new_bw / 2;
  new_freqs[1] = freq + new_bw / 2;

  new_start_channels[0] = start_channel;
  new_start_channels[1] = end_channel - new_nchan;

  new_end_channels[0] = start_channel + new_nchan;
  new_end_channels[1] = end_channel;

  // configure each format for a socket
  if (header1.set ("NPOL", "%u", new_npol) < 0)
    throw invalid_argument ("failed to write NPOL to header1");
  if (header1.set ("NCHAN", "%u", new_nchan) < 0)
    throw invalid_argument ("failed to write NCHAN to header1");
  if (header1.set ("FREQ", "%lf", new_freqs[0]) < 0)
    throw invalid_argument ("failed to write FREQ to header1");
  if (header1.set ("BW", "%lf", new_bw) < 0)
    throw invalid_argument ("failed to write BW to header1");
  if (header1.set ("START_CHANNEL", "%u", new_start_channels[0]) < 0)
    throw invalid_argument ("failed to write START_CHANNEL to header1");
  if (header1.set ("END_CHANNEL", "%u", new_end_channels[0]) < 0)
    throw invalid_argument ("failed to write END_CHANNEL to header1");

  if (header2.set ("NPOL", "%u", new_npol) < 0)
    throw invalid_argument ("failed to write NPOL to header2");
  if (header2.set ("NCHAN", "%u", new_nchan) < 0)
    throw invalid_argument ("failed to write NCHAN to header2");
  if (header2.set ("FREQ", "%lf", new_freqs[1]) < 0)
    throw invalid_argument ("failed to write FREQ to header2");
  if (header2.set ("BW", "%lf", new_bw) < 0)
    throw invalid_argument ("failed to write BW to header2");
  if (header2.set ("START_CHANNEL", "%u", new_start_channels[1]) < 0)
    throw invalid_argument ("failed to write START_CHANNEL to header2");
  if (header2.set ("END_CHANNEL", "%u", new_end_channels[1]) < 0)
    throw invalid_argument ("failed to write END_CHANNEL to header2");

  open (header1.raw(), header2.raw());

  return true;
}

// write the ascii header to the datablock, then
void spip::IBVReceiveMerge2DB::open (const char * header1_str, const char * header2_str)
{
  if (verbose)
    cerr << "spip::IBVReceiveMerge2DB::open()" << endl;

  // open the data block for writing  
  db1->open();
  db2->open();

  // write the header
  db1->write_header (header1_str);
  db2->write_header (header2_str);
}

void spip::IBVReceiveMerge2DB::close ()
{
  if (verbose)
    cerr << "spip::IBVReceiveMerge2DB::close()" << endl;
  if (db1->is_block_open())
  {
    if (verbose)
      cerr << "spip::IBVReceiveMerge2DB::close db1->close_block(" << db1->get_data_bufsz() << ")" << endl;
    db1->close_block(db1->get_data_bufsz());
  }

  if (db2->is_block_open())
  {
    if (verbose)
      cerr << "spip::IBVReceiveMerge2DB::close db2->close_block(" << db2->get_data_bufsz() << ")" << endl;
    db2->close_block(db2->get_data_bufsz());
  }

  // close the data block, ending the observation
  db1->close();
  db2->close();

  header1.reset();
  header2.reset();

  stats[0]->reset();
  stats[1]->reset();
}

void spip::IBVReceiveMerge2DB::send_terminal_packets()
{
#ifdef _DEBUG
  cerr << "spip::IBVReceiveMerge2DB::send_terminal_packets()" << endl;
#endif

  // create and open a UDP sending socket
  for (unsigned i=0; i<2; i++)
  {
    UDPSocketSend * sock = new UDPSocketSend();
    if (data_mcasts[i].size() > 0)
    {
      sock->open_multicast(data_mcasts[i], data_ports[i], data_hosts[i]);
    }
    else
    {
      sock->open (data_hosts[i], data_ports[i], data_hosts[i]);
    }
    sock->resize (32);
    sock->send();
    sock->close_me();
    delete sock;
  }
}


void spip::IBVReceiveMerge2DB::start_threads (int c1, int c2)
{
  // cpu cores on which to bind each recv thread
  cores[0] = c1;
  cores[1] = c2;

  // flag for whether the recv thread has filled the current buffer
  full[0] = true;
  full[1] = true;

  control_states[0] = Idle;
  control_states[1] = Idle;

  curr_blocks[0] = NULL;
  curr_blocks[1] = NULL;
  next_blocks[0] = NULL;
  next_blocks[1] = NULL;

  spip::IBVQueue::keep_receiving = true;

  pthread_create (&datablock_thread_id, NULL, datablock_thread_wrapper, this);
  pthread_create (&recv_thread1_id, NULL, recv_thread1_wrapper, this);
  sleep(2);
  pthread_create (&recv_thread2_id, NULL, recv_thread2_wrapper, this);
  pthread_create (&stats_thread_id, NULL, stats_thread_wrapper, this);
}

void spip::IBVReceiveMerge2DB::join_threads ()
{
  void * result;
  pthread_join (datablock_thread_id, &result);
  pthread_join (recv_thread1_id, &result);
  pthread_join (recv_thread2_id, &result);
  pthread_join (stats_thread_id, &result);
}

bool spip::IBVReceiveMerge2DB::datablock_thread ()
{
#ifdef _DEBUG
  cerr << "spip::IBVReceiveMerge2DB::datablock_thread starting" << endl;
#endif
  pthread_mutex_lock (&mutex_db);
  pthread_mutex_lock (&(mutex_recvs[0]));
  pthread_mutex_lock (&(mutex_recvs[1]));

  uint64_t ibuf = 0;

#ifdef _DEBUG
  cerr << "spip::IBVReceiveMerge2DB::datablock_thread locked mutexes" << endl;
#endif

  // wait for the starting command from the control_thread
  while (control_cmd == None)
    pthread_cond_wait (&cond_db, &mutex_db);

#ifdef _DEBUG
    cerr << "spip::IBVReceiveMerge2DB::datablock control_cmd  != None" << endl;
#endif

  // if we have a start command then we can continue
  if (control_cmd == Start)
  {
   // open the data block for writing
    curr_blocks[0] = (char *) (db1->open_block());
    curr_blocks[1] = (char *) (db2->open_block());
    next_blocks[0] = (char *) (db1->open_block());
    next_blocks[1] = (char *) (db2->open_block());

    // state of this thread
    control_state = Active;

    full[0] = full[1] = false;

#ifdef _DEBUG
    cerr << "spip::IBVReceiveMerge2DB::datablock opened buffers for ibuf=" << ibuf << endl;
#endif
  }
  else if (control_cmd == Stop || control_cmd == Quit)
  {
    control_state = Stopping;
#ifdef _DEBUG
    cerr << "spip::IBVReceiveMerge2DB::datablock_thread received "
         <<  "a Stop command prior to starting" << endl;
#endif
  }
  else
  {
    throw invalid_argument ("datathread encounter an unexpected control_cmd");
  }

  pthread_mutex_unlock (&mutex_db);

  // signal receive threads to wake up and inspect control_state
  control_states[0] = control_states[1] = control_state;
  pthread_cond_signal (&(cond_recvs[0]));
  pthread_cond_signal (&(cond_recvs[1]));
  pthread_mutex_unlock (&(mutex_recvs[0]));
  pthread_mutex_unlock (&(mutex_recvs[1]));

#ifdef _DEBUG
  cerr << "spip::IBVReceiveMerge2DB::datablock signaled recv " << ibuf << endl;
#endif

  // while the receiving state is Active
  while (control_state == Active)
  {
    // zero the next buffer that DB would provide
    //db1->zero_next_block();
    //db2->zero_next_block();

    // wait for RECV threads to fill buffer
    pthread_mutex_lock (&mutex_db);

    // if the current buffer has been filled  by both receive threads
    while (!full[0] || !full[1])
    {
#ifdef _DEBUG
      cerr << "spip::IBVReceiveMerge2DB::datablock checking buffer " << ibuf 
           << " [" << full[0] << "," << full[1] << "]" << endl;
#endif
      pthread_cond_wait (&cond_db, &mutex_db);
    }
    
#ifdef _DEBUG
    cerr << "spip::IBVReceiveMerge2DB::datablock filled buffer " << ibuf << endl;
#endif
    
    // close curr data blocks
    db1->close_block(db1->get_data_bufsz());
    db2->close_block(db2->get_data_bufsz());

    // acquire both recv threads' mutexes
    pthread_mutex_lock (&(mutex_recvs[0]));
    pthread_mutex_lock (&(mutex_recvs[1]));

    ibuf++;

    // check for state changes
    if (control_cmd == Stop || control_cmd == Quit)
    {
      cerr << "STATE=Idle" << endl;
      control_state = Idle;
      control_states[0] = control_states[1] = control_state;

      // close next data blocks TODO change to 0 bytes?
      db1->close_block(db1->get_data_bufsz());
      db2->close_block(db2->get_data_bufsz());
    }
    else
    {
      // switch the curr/next blocks
      curr_blocks[0] = next_blocks[0];  
      curr_blocks[1] = next_blocks[1];  

      // open new next blocks
      next_blocks[0] = (char *) (db1->open_block());
      next_blocks[1] = (char *) (db2->open_block());

      full[0] = full[1] = false;
    }

    // release the DB mutex now that we have the recv threads locked
    pthread_mutex_unlock (&mutex_db);
    pthread_cond_signal (&(cond_recvs[0]));
    pthread_cond_signal (&(cond_recvs[1]));
    pthread_mutex_unlock (&(mutex_recvs[0]));
    pthread_mutex_unlock (&(mutex_recvs[1]));
  }

  close();

#ifdef _DEBUG
  cerr << "spip::IBVReceiveMerge2DB::data_block exiting" << endl;
#endif

  return true;
}

bool spip::IBVReceiveMerge2DB::receive_thread (int p)
{
#ifdef _DEBUG
  cerr << "spip::IBVReceiveMerge2DB::receive_thread[" << p << "] starting" << endl;
#endif
#ifdef HAVE_HWLOC
  spip::HardwareAffinity hw_affinity;
  hw_affinity.bind_thread_to_cpu_core (cores[p]);
  hw_affinity.bind_to_memory (cores[p]);
#endif

  // allocated and configured in main threasd
  UDPFormat * format = formats[p];
  UDPStats * stat = stats[p];
  IBVQueue * queue = queues[p];

  pthread_cond_t * cond_recv = &(cond_recvs[p]);
  pthread_mutex_t * mutex_recv = &(mutex_recvs[p]);

  // block accounting 
  const int64_t data_bufsz = (db1->get_data_bufsz() + db2->get_data_bufsz()) / 2;
  int64_t curr_byte_offset;
  int64_t next_byte_offset = 0;
  int64_t last_byte_offset = data_bufsz;

  // sub-band accounting
  const int64_t pol_stride = format->get_resolution();
  const int64_t subband_stride = pol_stride / 2;

  int64_t pol_bytes_curr_buf = 0;
  int64_t pol_bytes_next_buf = 0;
  int64_t byte_offset;
  uint64_t pol_subband_offset = p * pol_stride / 2;

  bool filled_curr_buffer = false;
  unsigned bytes_received;
  int got;
  uint64_t ibuf = 0;

#ifdef _DEBUG
  cerr << "spip::IBVReceiveMerge2DB::receive_thread[" << p << "] trying to lock mutex" << endl;
#endif
  // wait for datablock thread to change state to Active
  pthread_mutex_lock (mutex_recv);

  // wait for start command
  while (control_states[p] == Idle)
    pthread_cond_wait (cond_recv, mutex_recv);
  if (verbose)
    cerr << "spip::IBVReceiveMerge2DB::receive["<<p<<"] control_state != Idle" << endl;
  pthread_mutex_unlock (mutex_recv);

  // main data acquisition loop
  while (control_states[p] == Active)
  {
    // wait until the datablock thread sets the state of this buffer
    // to empty (i.e. when it has provided a new buffer to fill
    pthread_mutex_lock (mutex_recv);

    while (full[p] && control_states[p] == Active)
    {
#ifdef _DEBUG
      cerr << "spip::IBVReceiveMerge2DB::receive["<<p<<"] waiting for empty " << ibuf << " [" << full[0] << ", " << full[1] << "]" << endl;
#endif
      pthread_cond_wait (cond_recv, mutex_recv);
    }

    if (control_states[p] == Active)
    {
      curr_byte_offset = next_byte_offset;
      next_byte_offset += data_bufsz;
      last_byte_offset += data_bufsz;

#ifdef _DEBUG
      if (p == 1)
        cerr << "spip::IBVReceiveMerge2DB::receive["<<p<<"] filling buffer " 
             << ibuf << " [" <<  curr_byte_offset << " - " << next_byte_offset
             << " - " << last_byte_offset << "]" << endl;
#endif

      // signal other threads waiting on the condition
      pthread_mutex_unlock (mutex_recv);
      filled_curr_buffer = false;
      
      // account for data saved into the next buf
      pol_bytes_curr_buf = pol_bytes_next_buf;

      // while we have not filled this buffer with data from
      // this polarisation
      while (!filled_curr_buffer && spip::IBVQueue::keep_receiving)
      {
        // get a packet from the socket
        got = queue->open_packet();
        if (got > 0)
        {
          // byte offset is correct for 1 polarisation
          byte_offset = format->decode_packet ((char *)queue->buf_ptr, &bytes_received);

          if (byte_offset < 0)
          {
            // ignore if byte_offset is -ve
            queue->close_packet();

            // data stream is finished
            if (byte_offset == -2)
            {
              set_control_cmd(Stop);
            } 
          }
          // packet that is part of this observation
          else
          {
            // determine which sub-band this byte_offset resides in
            int64_t subband = format->get_subband (byte_offset, 2);

            // packet belongs in current buffer
            if ((byte_offset >= curr_byte_offset) && (byte_offset < next_byte_offset))
            {
              // block offset is same for a single pol or half-band dual pol
              // add offset due to polarisation (for the sub-band)
              int64_t block_offset = (byte_offset - curr_byte_offset) + pol_subband_offset;
              if (subband)
                block_offset -= subband_stride;

              pol_bytes_curr_buf += bytes_received;
              stat->increment_bytes (bytes_received);
              format->insert_last_packet (curr_blocks[subband] + block_offset);
              queue->close_packet();
            }

            // packet fits in the next block
            else if ((byte_offset >= next_byte_offset) && (byte_offset < last_byte_offset))
            {
              int64_t block_offset = (byte_offset - next_byte_offset) + pol_subband_offset;
              if (subband)
                block_offset -= subband_stride;

              pol_bytes_next_buf += bytes_received;
              stat->increment_bytes (bytes_received);
              format->insert_last_packet (next_blocks[subband] + block_offset);
              queue->close_packet();
            }

            // packet belong to a previous buffer (this is a drop that has already been counted)
            else if (byte_offset < curr_byte_offset)
            {
              queue->close_packet();
            }

            // packet belongs to a future buffer, that is beyond the next_block
            else
            {
              filled_curr_buffer = true;
            }
          }

          // close open data block buffer if is is now full
          if (pol_bytes_curr_buf >= data_bufsz || filled_curr_buffer)
          {
#ifdef _DEBUG
            if (p == 1)
              cerr << "spip::IBVReceiveMerge2DB::receive["<<p<<"] close_block "
                   << " pol_bytes_curr_buf=" << pol_bytes_curr_buf 
                   << " data_bufsz=" << data_bufsz 
                   << " filled_curr_buffer=" << filled_curr_buffer << endl;
#endif
            stat->dropped_bytes (data_bufsz - pol_bytes_curr_buf);
            filled_curr_buffer = true;
          }
        }
        else if (got == 0)
        {
          set_control_cmd (Stop);
        }
        else
        {
        }
      }

      pthread_mutex_lock (&mutex_db);
      full[p] = true;

#ifdef _DEBUG
      cerr << "spip::IBVReceiveMerge2DB::receive["<<p<<"] filled buffer " << ibuf << endl; 
#endif
      ibuf++;
      pthread_cond_signal (&cond_db);
      pthread_mutex_unlock (&mutex_db);
    }
    else
      pthread_mutex_unlock (mutex_recv);
  }

#ifdef _DEBUG
  cerr << "spip::IBVReceiveMerge2DB::receive["<<p<<"] exiting" << endl;
#endif

  if (control_states[p] == Idle)
    return true;
  else
    return false;
}

/* 
 *  Thread to print simple capture statistics
 */
void spip::IBVReceiveMerge2DB::stats_thread()
{
  uint64_t b_recv_total[2] = {0, 0};
  uint64_t b_recv_curr[2];
  uint64_t b_recv_1sec;

  uint64_t s_total[2] = {0, 0};
  uint64_t s_curr[2];
  //uint64_t s_1sec;

  uint64_t b_drop_total[2] = {0,0};
  uint64_t b_drop_curr[2];
  uint64_t b_drop_1sec;

  float gb_recv_ps[2] = {0, 0};
  float mb_recv_ps[2] = {0, 0};
  float gb_drop_ps[2] = {0, 0};

#ifdef _DEBUG
  cerr << "spip::IBVReceiveMerge2DB::stats_thread starting polling" << endl;
#endif

  while (control_cmd != Stop && control_cmd != Quit)
  {
    // reset statistics
    b_recv_total[0] = b_recv_total[1] = 0;
    b_drop_total[0] = b_drop_total[1] = 0;
    s_total[0] = s_total[1] = 0;

    while (control_state == Active)
    {
      for (unsigned i=0; i<2; i++)
      {
        // get a snapshot of the data as quickly as possible
        b_recv_curr[i] = stats[i]->get_data_transmitted();
        b_drop_curr[i] = stats[i]->get_data_dropped();
        s_curr[i] = stats[i]->get_nsleeps();

        // calc the values for the last second
        b_drop_1sec = b_drop_curr[i] - b_drop_total[i];
        b_recv_1sec = b_recv_curr[i] - b_recv_total[i];
        //s_1sec = s_curr[i] - s_total[i];

        // update the totals
        b_drop_total[i] = b_drop_curr[i];
        b_recv_total[i] = b_recv_curr[i];
        s_total[i] = s_curr[i];

        gb_drop_ps[i] = (double) (b_drop_1sec * 8) / 1000000000;
        mb_recv_ps[i] = (double) b_recv_1sec / 1000000;
        gb_recv_ps[i] = (mb_recv_ps[i] * 8)/1000;
      }

      // determine how much memory is free in the receivers
      fprintf (stderr,"Recv %6.3f (%6.3f, %6.3f) [Gb/s] Dropped %6.3f (%6.3f + %6.3f) [Gb/s] Total %lu B\n", 
               gb_recv_ps[0] + gb_recv_ps[1], gb_recv_ps[0], gb_recv_ps[1],
               gb_drop_ps[0] + gb_drop_ps[1], gb_drop_ps[0], gb_drop_ps[1],
               b_drop_total[0] + b_drop_total[1]);
      sleep (1);
    }
    sleep(1);
  }
#ifdef _DEBUG
  cerr << "spip::IBVReceiveMerge2DB::stats_thread exiting";
#endif
}

