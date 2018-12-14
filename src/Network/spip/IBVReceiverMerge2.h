
#ifndef __IBVReceiverMerge2_h
#define __IBVReceiverMerge2_h

#include "dada_def.h"

#include "config.h"

#include "spip/AsciiHeader.h"
#include "spip/IBVQueue.h"
#include "spip/UDPFormat.h"
#include "spip/UDPStats.h"

#include <iostream>
#include <cstdlib>
#include <pthread.h>

#define NEW_LOGIC

namespace spip {

  enum ControlCmd   { None, Start, Stop, Quit };
  enum ControlState { Idle, Active, Stopping };

  class IBVReceiverMerge2 {

    public:

      IBVReceiverMerge2 (boost::asio::io_service& io_service);

      ~IBVReceiverMerge2 ();

      int configure (const char * config);

      void set_formats (UDPFormat * fmt1, UDPFormat * fmt2);

      void set_control_cmd (ControlCmd cmd);

      void start_control_thread (int port);
      void stop_control_thread ();

      static void * control_thread_wrapper (void * obj)
      {
        ((IBVReceiverMerge2*) obj )->control_thread ();
        pthread_exit (NULL);
      }

      void start_threads (int core1, int core2);

      void join_threads ();

      static void * datablock_thread_wrapper (void * obj)
      {
        ((IBVReceiverMerge2*) obj )->datablock_thread ();
        pthread_exit (NULL);
      }

      bool datablock_thread ();

      static void * recv_thread1_wrapper (void * obj)
      {
        ((IBVReceiverMerge2*) obj )->receive_thread (0);
        pthread_exit (NULL);
      }

      static void * recv_thread2_wrapper (void * obj)
      {
        ((IBVReceiverMerge2*) obj )->receive_thread (1);
        pthread_exit (NULL);
      }

      bool receive_thread (int pol);

      static void * stats_thread_wrapper (void * obj)
      {
        ((IBVReceiverMerge2*) obj )->stats_thread ();
        pthread_exit (NULL);
      }

      void stats_thread();

      bool open ();

      void open (const char * header1, const char * header2);

      void close ();

      void send_terminal_packets();

      bool receive ();

    protected:

      void control_thread ();

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

      IBVQueue queue1;

      IBVQueue queue2;

      IBVQueue * queues[2];

      UDPFormat * formats[2];

      UDPStats * stats[2];

      int cores[2];

#ifdef NEW_LOGIC
      int64_t full[2];
#else
      bool full[2];
#endif

      char * curr_blocks[2];
      char * next_blocks[2];
      char * last_blocks[2];

      char * curr_blocks_ptrs[2][2];
      char * next_blocks_ptrs[2][2];

      char * tmp_blocks[2];

      unsigned chunk_size;

      uint64_t timestamp;

      AsciiHeader config;

      AsciiHeader header1;

      AsciiHeader header2;

      uint64_t block_bufsz;
  };

}

#endif
