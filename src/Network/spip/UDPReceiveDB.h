
#ifndef __UDPReceiveDB_h
#define __UDPReceiveDB_h

#include "config.h"

#include "spip/AsciiHeader.h"
#include "spip/UDPSocketReceive.h"
#include "spip/UDPFormat.h"
#include "spip/UDPStats.h"
#include "spip/DataBlockWrite.h"

#include <iostream>
#include <cstdlib>
#include <pthread.h>

#ifdef  HAVE_VMA
#include <mellanox/vma_extra.h>
#endif

namespace spip {

  enum ControlCmd   { None, Start, Stop, Quit };
  enum ControlState { Idle, Active, Stopping };

  class UDPReceiveDB {

    public:

      UDPReceiveDB (const char * key_string);

      ~UDPReceiveDB ();

      int configure (const char * config);

      void set_format (UDPFormat * fmt);

      void set_control_cmd (ControlCmd cmd);

      void start_control_thread (int port);

      void stop_control_thread ();

      static void * control_thread_wrapper (void * obj)
      {
        // ensure the control thread is not offloaded
#ifdef HAVE_VMA
        pthread_t id = pthread_self();
        struct vma_api_t * vma_api = vma_get_api();
        if (vma_api)
        {
          vma_api->thread_offload (0, id);
        }
#endif
        ((UDPReceiveDB*) obj)->control_thread ();
        pthread_exit (NULL);
      }

      bool open ();

      void open (const char * header);

      void close ();

      bool receive (int core);

      void start_capture () { control_cmd = Start; };

      void stop_capture () { control_cmd = Stop; };

      static void * stats_thread_wrapper (void * obj)
      {
        ((UDPReceiveDB*) obj )->stats_thread ();
        pthread_exit (NULL);
      }

      void start_stats_thread ();

      void stop_stats_thread ();

      void stats_thread ();

      UDPStats * get_stats () { return stats; };

      uint64_t get_data_bufsz () { return db->get_data_bufsz(); };

    protected:

      void control_thread ();

      std::string data_host;

      std::string data_mcast;

      int data_port;

      UDPSocketReceive * sock;

      UDPFormat * format;

      UDPStats * stats;

      pthread_t stats_thread_id;

      DataBlockWrite * db;

      pthread_t control_thread_id;

      int control_port;

      ControlCmd control_cmd;

      ControlState control_state;

      AsciiHeader config;

      AsciiHeader header;

#ifdef HAVE_VMA
      struct vma_api_t *vma_api;

      struct vma_packets_t* pkts;
#else
      char vma_api;
#endif

      bool keep_receiving;

      unsigned nchan;

      unsigned ndim;

      unsigned nbit;

      unsigned npol;

      float bw;

      float channel_bw;

      float tsamp;

      unsigned bits_per_second;

      unsigned bytes_per_second;

      uint64_t b_recv_curr;
      uint64_t b_drop_curr;
      uint64_t s_curr;
      uint64_t b_recv_total;
      uint64_t b_drop_total;
      uint64_t s_total;

      double bytes_recv_ps;
      double bytes_drop_ps;
      double sleeps_ps;

      struct timeval curr;
      struct timeval prev;

      char verbose;

    private:

      int core;

      pthread_cond_t cond;

      pthread_mutex_t mutex;

  };

}

#endif
