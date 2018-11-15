
#ifndef __IBVReceiveDB_h
#define __IBVReceiveDB_h

#include "config.h"

#include "spip/AsciiHeader.h"
#include "spip/IBVQueue.h"
#include "spip/UDPFormat.h"
#include "spip/UDPStats.h"
#include "spip/UDPOverflow.h"
#include "spip/DataBlockWrite.h"

#include <iostream>
#include <cstdlib>
#include <pthread.h>

namespace spip {

  enum ControlCmd   { None, Record, Monitor, Stop, Quit };
  enum ControlState { Idle, Recording, Monitoring, Stopping, Quitting };

  class IBVReceiveDB {

    public:

      IBVReceiveDB (const char * key_string, boost::asio::io_service& io_service);

      ~IBVReceiveDB ();

      void set_verbosity (int v) { verbose = v; };

      int configure (const char * config);

      void set_format (UDPFormat * fmt);

      void set_control_cmd (ControlCmd cmd);

      void start_control_thread (int port);

      void stop_control_thread ();

      static void * control_thread_wrapper (void * obj)
      {
        ((IBVReceiveDB*) obj)->control_thread ();
        pthread_exit (NULL);
      }

      bool open ();

      void open (const char * header);

      void close ();

      bool receive (int core);

      void receive_block (UDPFormat * fmt);

      void start_capture () { set_control_cmd (Record); };

      void stop_capture () { set_control_cmd (Stop); };

      void quit_capture () { set_control_cmd (Quit); };

      static void * stats_thread_wrapper (void * obj)
      {
        ((IBVReceiveDB*) obj )->stats_thread ();
        pthread_exit (NULL);
      }

      void start_stats_thread ();

      void stop_stats_thread ();

      void stats_thread ();

      UDPStats * get_stats () { return udp_stats; };

      uint64_t get_data_bufsz () { return db->get_data_bufsz(); };

    protected:

      void control_thread ();

      std::string data_host;

      std::string data_mcast;

      int data_port;

      IBVQueue queue;

      UDPFormat * format;

      UDPStats * udp_stats;

      UDPOverflow * overflow;

      pthread_t stats_thread_id;

      DataBlockWrite * db;

      pthread_t control_thread_id;

      int control_port;

      ControlCmd control_cmd;

      ControlState control_state;

      AsciiHeader config;

      AsciiHeader header;

      uint64_t resolution;

      char verbose;

      char * block;

      char * overflow_block;

      pthread_cond_t cond;

      pthread_mutex_t mutex;

    private:


      int64_t curr_byte_offset;

      int64_t next_byte_offset;

      int64_t overflow_maxbyte;

      uint64_t data_block_bufsz;

      size_t packet_size;

      size_t header_size;

      size_t buffer_size;

      int core;

  };

}

#endif
