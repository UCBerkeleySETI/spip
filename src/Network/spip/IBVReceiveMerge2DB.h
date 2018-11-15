
#ifndef __IBVReceiveMerge2DB_h
#define __IBVReceiveMerge2DB_h

#include "dada_def.h"

#include "config.h"

#include "spip/AsciiHeader.h"
#include "spip/IBVQueue.h"
#include "spip/UDPFormat.h"
#include "spip/UDPStats.h"
#include "spip/DataBlockWrite.h"

#include <iostream>
#include <cstdlib>
#include <pthread.h>

namespace spip {

  enum ControlCmd   { None, Start, Stop, Quit };
  enum ControlState { Idle, Active, Stopping };

  class IBVReceiveMerge2DB {

    public:

      IBVReceiveMerge2DB (const char * key1, const char * key2);

      ~IBVReceiveMerge2DB ();

      int configure (const char * config);

      void set_formats (UDPFormat * fmt1, UDPFormat * fmt2);

      void set_control_cmd (ControlCmd cmd);

      void start_control_thread (int port);
      void stop_control_thread ();

      static void * control_thread_wrapper (void * obj)
      {
        ((IBVReceiveMerge2DB*) obj )->control_thread ();
        pthread_exit (NULL);
      }

      void start_threads (int core1, int core2);

      void join_threads ();

      static void * datablock_thread_wrapper (void * obj)
      {
        ((IBVReceiveMerge2DB*) obj )->datablock_thread ();
        pthread_exit (NULL);
      }

      bool datablock_thread ();

      static void * recv_thread1_wrapper (void * obj)
      {
        ((IBVReceiveMerge2DB*) obj )->receive_thread (0);
        pthread_exit (NULL);
      }

      static void * recv_thread2_wrapper (void * obj)
      {
        ((IBVReceiveMerge2DB*) obj )->receive_thread (1);
        pthread_exit (NULL);
      }

      bool receive_thread (int pol);

      static void * stats_thread_wrapper (void * obj)
      {
        ((IBVReceiveMerge2DB*) obj )->stats_thread ();
        pthread_exit (NULL);
      }

      void stats_thread();

      bool open ();

      void open (const char * header1, const char * header2);

      void close ();

      void send_terminal_packets();

      bool receive ();

      uint64_t get_data_bufsz () { return db1->get_data_bufsz(); };

    protected:

      void control_thread ();

      DataBlockWrite * db1;

      DataBlockWrite * db2;

      int control_port;

      ControlCmd control_cmd;

      ControlState control_state;

      ControlState control_states[2];

      std::string data_hosts[2];

      int data_ports[2];

      std::string data_mcasts[2];

      unsigned nchan;

      unsigned ndim;

      unsigned nbit;

      unsigned npol;

      double bw;

      double tsamp;
      
      double freq;

      unsigned start_channel;

      unsigned end_channel;

      uint64_t resolution;

      unsigned bits_per_second;

      unsigned bytes_per_second;

      int64_t start_adc_sample;

      char verbose;

    private:
  
      pthread_t control_thread_id;

      pthread_t datablock_thread_id;

      pthread_t recv_thread1_id;

      pthread_t recv_thread2_id;

      pthread_t stats_thread_id;

      pthread_cond_t cond_db;

      pthread_mutex_t mutex_db;

      pthread_cond_t cond_recvs[2];

      pthread_mutex_t mutex_recvs[2];

      IBVQueue * queues[2];

      UDPFormat * formats[2];

      UDPStats * stats[2];

      int cores[2];

      bool full[2];

      char * curr_blocks[2];

      char * next_blocks[2];

      unsigned chunk_size;

      uint64_t timestamp;

      AsciiHeader config;

      AsciiHeader header1;

      AsciiHeader header2;

  };

}

#endif
