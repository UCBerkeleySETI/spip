/***************************************************************************
 *
 *   Copyright (C) 2017 Andrew Jameson
 *   Licensed under the Academic Free License version 2.1
 *
 ***************************************************************************/

#include "config.h"

#include "spip/Error.h"
#ifdef HAVE_VMA
#include "spip/UDPSocketReceiveVMA.h"
#else
#include "spip/UDPSocketReceive.h"
#endif
#include "spip/UDPSocketSend.h"
#include "spip/HardwareAffinity.h"

#ifdef HAVE_CUDA
#include "spip/keckrtc_kernels.h"
#endif
#include "spip/KeckRTCDefs.h"

#include <unistd.h>
#include <signal.h>
#include <pthread.h>

#include <cstdio>
#include <cstring>
#include <iostream>
#include <stdexcept>

void usage();
void signal_handler (int signal_value);
char quit_threads = 0;

using namespace std;

int main(int argc, char *argv[]) try
{
  string server;

  int port;

  string client;

  spip::HardwareAffinity hw_affinity;

  int verbose = 0;

#ifdef HAVE_CUDA
  int device_id = 0;
#endif

  int core = -1;

  opterr = 0;
  int c;

#ifdef HAVE_CUDA
  while ((c = getopt(argc, argv, "b:d:hv")) != EOF) 
#else
  while ((c = getopt(argc, argv, "b:hv")) != EOF) 
#endif
  {
    switch(c) 
    {
      case 'b':
        core = atoi(optarg);
        break;

#ifdef HAVE_CUDA
      case 'd':
        device_id = atoi(optarg);
        break;
#endif
  
      case 'h':
        cerr << "Usage: " << endl;
        usage();
        exit(EXIT_SUCCESS);
        break;

      case 'v':
        verbose++;
        break;

      default:
        cerr << "Unrecognised option [" << c << "]" << endl;
        usage();
        return EXIT_FAILURE;
       break;
    }
  }

  // check command line arguments
  int nargs = argc - optind;
  if (nargs != 3)
  {
    cerr << "Error: 3 command line arguments are required" << endl;
    usage();
    return (EXIT_FAILURE);
  }

  if (core >= 0)
  {
    hw_affinity.bind_process_to_cpu_core (core);
    //hw_affinity.bind_to_memory (core);
  }

#ifdef HAVE_CUDA
  int device_count = 0;
  cudaError_t err = cudaGetDeviceCount(&device_count);
  if (err != cudaSuccess)
    throw runtime_error ("cudaGetDeviceCount failed");

  if (device_id >= device_count)
    throw runtime_error ("Specified CUDA device not available");

  err = cudaSetDevice (device_id);
  if (err != cudaSuccess)
    throw runtime_error ("cudaSetDevice failed");

  cudaStream_t stream;
  err = cudaStreamCreate(&stream);
   if (err != cudaSuccess)
    throw runtime_error ("cudaStreamCreate failed");

#endif

  server = std::string(argv[optind+0]);
  port = atoi(argv[optind+1]);
  client = std::string(argv[optind+2]);

  // UDP packet size for send/recv
  size_t bufsz = KeckRTC_UDP_SIZE;

  // create a UDP receiving socket
#ifdef HAVE_VMA
  spip::UDPSocketReceiveVMA sock_recv;
#else
  spip::UDPSocketReceive sock_recv;
#endif
  cerr << "opening recv socket on " << server << ":" << port << endl;
  sock_recv.open (server, port);
  sock_recv.resize (bufsz);
  sock_recv.set_block ();

  port++;

  spip::UDPSocketSend sock_send;
  cerr << "opening send socket to " << client << ":" << port << " from  " << server << endl;
  sock_send.open (client, port, server);
  sock_send.resize (bufsz);

  void * send_buf_ptr = (void *) sock_send.get_buf();
  void * recv_buf_ptr = (void *) sock_recv.get_buf();

#ifdef HAVE_CUDA
  unsigned int flags = 0;
  cudaError_t rval;

  // register the udp socket buffers as host memory
  rval = cudaHostRegister (send_buf_ptr, bufsz, flags);
  if (rval != cudaSuccess)
    cerr << "cudaHostRegister failed on sock_send" << endl;

  rval = cudaHostRegister (recv_buf_ptr, bufsz, flags);
  if (rval != cudaSuccess)
    cerr << "cudaHostRegister failed on sock_recv" << endl;

  // allocat host memory
  void * host_buf, * dev_buf;
  rval = cudaMallocHost (&host_buf, KeckRTC_HEAP_SIZE);
  if (rval != cudaSuccess)
    cerr << "cudaMallocHost failed on host_buf" << endl;

  // allocate device memory
  rval = cudaMalloc (&dev_buf, KeckRTC_HEAP_SIZE);
  if (rval != cudaSuccess)
    cerr << "cudaMalloc failed on dev_buf" << endl;
#else
  void * host_buf = malloc (KeckRTC_HEAP_SIZE);
#endif

  size_t reply_size;
  bool keep_receiving = true;

  char * host_ptr = (char *) host_buf;

  while (keep_receiving)
  {
    uint64_t bytes_received = 0;
    while (keep_receiving && bytes_received < KeckRTC_HEAP_SIZE)
    {
      // receive the packet
      reply_size = sock_recv.recv_from ();
      sock_recv.consume_packet();
      if (reply_size == KeckRTC_UDP_SIZE)
        memcpy (host_ptr + bytes_received, recv_buf_ptr, reply_size);
      else
        keep_receiving = false;
      bytes_received += reply_size;
    }

#ifdef HAVE_CUDA
    rval = cudaMemcpyAsync (dev_buf, host_ptr, KeckRTC_HEAP_SIZE, cudaMemcpyHostToDevice, stream);
    if (rval != cudaSuccess)
      cerr << "cudaMemcpyAsync failed on sock_recv" << endl;

    // perform some sort of operation 
    keckrtc_dummy (dev_buf, bufsz, stream);

    rval = cudaMemcpyAsync (send_buf_ptr, dev_buf, bufsz, cudaMemcpyDeviceToHost, stream);
    if (rval != cudaSuccess)
      cerr << "cudaMemcpyAsync failed on sock_recv" << endl;

    rval = cudaStreamSynchronize (stream);
      if (rval != cudaSuccess)
      cerr << "cudaStreamSynchronize failed" << endl;
#endif

    // send a reply
    sock_send.send (bufsz);
  }

#ifdef HAVE_CUDA
  rval = cudaHostUnregister (send_buf_ptr);
  if (rval != cudaSuccess)
    cerr << "cudaHostUnregister failed on sock_send" << endl;

  rval = cudaHostUnregister (recv_buf_ptr);
  if (rval != cudaSuccess)
    cerr << "cudaHostUnregister failed on sock_recv" << endl;

  cudaFree (dev_buf);
  cudaFreeHost (host_buf);

#endif

  sock_send.close_me();
  sock_recv.close_me();

  return 0;
}
catch (std::exception& exc)
{
  cerr << "ERROR: " << exc.what() << endl;
  return -1;
  return 0;
}
catch (Error& error)
{
  cerr << error << endl;
  return -1;
}


void usage() 
{
  cout << "keckrtc_server [options] server port server\n"
    "  server      server hostname/IP address\n"
    "  port        port for send and receiver\n"
    "  server      server hostname/IP address\n"
    "  -b core     bind computation to specified CPU core\n"
    "  -h          print this help text\n"
    "  -v          verbose output\n"
    << endl;
}

/*
 *  Simple signal handler to exit more gracefully
 */
void signal_handler(int signalValue)
{
  fprintf(stderr, "received signal %d\n", signalValue);
  if (quit_threads) 
  {
    fprintf(stderr, "received signal %d twice, hard exit\n", signalValue);
    exit(EXIT_FAILURE);
  }
  quit_threads = 1;
}
