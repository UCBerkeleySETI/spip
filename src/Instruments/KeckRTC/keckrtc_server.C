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
#include "spip/KeckRTCUtil.h"

#include <float.h>
#include <unistd.h>
#include <fcntl.h>
#include <signal.h>
#include <pthread.h>
#include <sys/time.h>

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

  unsigned frame_size = KeckRTC_HEAP_SIZE;

  unsigned frame_rate = 2000;

  unsigned duration = 10;

  bool process_data = true;

#ifdef HAVE_CUDA
  int device_id = 0;
#endif

  int core = -1;

  opterr = 0;
  int c;

#ifdef HAVE_CUDA
  while ((c = getopt(argc, argv, "b:d:f:g:hir:v")) != EOF) 
#else
  while ((c = getopt(argc, argv, "b:d:f:hir:v")) != EOF) 
#endif
  {
    switch(c) 
    {
      case 'b':
        core = atoi(optarg);
        break;

      case 'd':
        duration = atoi(optarg);
        break;

#ifdef HAVE_CUDA
      case 'g':
        device_id = atoi(optarg);
        break;
#endif

      case 'f':
        frame_size = atoi(optarg);
        break;

      case 'h':
        cerr << "Usage: " << endl;
        usage();
        exit(EXIT_SUCCESS);
        break;

      case 'i':
        process_data = false;
        break;

      case 'r':
        frame_rate = atoi(optarg);
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
  rval = cudaMallocHost (&host_buf, frame_size);
  if (rval != cudaSuccess)
    cerr << "cudaMallocHost failed on host_buf" << endl;

  // allocate device memory
  rval = cudaMalloc (&dev_buf, frame_size);
  if (rval != cudaSuccess)
    cerr << "cudaMalloc failed on dev_buf" << endl;
#else
  void * host_buf = malloc (frame_size);
#endif

  size_t reply_size;
  bool keep_receiving = true;

  char * host_ptr = (char *) host_buf;
  struct timeval start_frame;
  struct timeval end_frame;

  uint64_t frames_to_receive = frame_rate * duration;
  double * times = (double *) malloc (frames_to_receive * sizeof(double));

  cerr << "Ready to receive" << endl;
  uint64_t iframe = 0;

  while (keep_receiving)
  {
    uint64_t bytes_received = 0;
    while (keep_receiving && bytes_received < frame_size)
    {
      // receive the packet
      if (verbose)
        cerr << "Receiving " << KeckRTC_UDP_SIZE << " bytes" << endl;
      reply_size = sock_recv.recv_from ();
      sock_recv.consume_packet();
      if (reply_size == KeckRTC_UDP_SIZE)
      {
        if (process_data) 
          memcpy (host_ptr + bytes_received, recv_buf_ptr, reply_size);
      }
      else
        keep_receiving = false;
      bytes_received += reply_size;
    }

    gettimeofday (&start_frame, 0);

#ifdef HAVE_CUDA
    if (process_data) 
    {
    rval = cudaMemcpyAsync (dev_buf, host_ptr, frame_size, cudaMemcpyHostToDevice, stream);
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
    }
#endif

    gettimeofday (&end_frame, 0);

    double frame_time = diff_time (start_frame, end_frame);
    times[iframe] = frame_time;
    iframe++;

    if (verbose)
      cerr << "Sending reply of " << bufsz << " bytes" << endl;

    // send a reply
    sock_send.send (bufsz);
  }

  if (iframe > 0)
  { 
    uint64_t frames_sent = iframe - 1;
    
    double duration_min = DBL_MAX;
    double duration_max = -DBL_MAX;
    double duration_sum = 0;
      
    int flags = O_WRONLY | O_CREAT | O_TRUNC; 
    int perms = S_IRUSR | S_IWUSR | S_IRGRP | S_IROTH;
    int fd = open ("recv_timing.dat", flags, perms);
    ::write (fd, times, frames_sent * sizeof(double));
    ::close (fd);
     
    double time_sum = 0;
      
    for (iframe=0; iframe<frames_sent; iframe++)
    { 
      double duration = times[iframe];
        
      if (duration < duration_min)
        duration_min = duration;
      if (duration > duration_max)
        duration_max = duration;
      duration_sum += duration;
       
      time_sum += duration;
    }
      
    double duration_mean = duration_sum / double(frames_sent);
    cerr << "duration timing:" << endl;
    cerr << "  minimum=" << duration_min << " us" << endl;
    cerr << "  mean="    << duration_mean << " us" << endl;
    cerr << "  maximum=" << duration_max << " us" << endl;
     
    double bytes_per_microsecond = (frames_sent * frame_size) / time_sum;
    double gb_per_second = (bytes_per_microsecond * 1000000) / 1000000000;
      
    double frames_per_microsecond = frames_sent / time_sum;
    double frames_per_second = frames_per_microsecond *1e6;
      
    cerr << "  data_rate=" << gb_per_second << " Gb/s" << endl;
    cerr << "  frame_rate=" << frames_per_second << endl;
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
    "  -d secs     duration of test is seconds [default 10]\n"
    "  -f bytes    size of a frame in bytes [default "<< KeckRTC_HEAP_SIZE << "]\n"
#ifdef HAVE_CUDA
    "  -g id       use CUDA device id\n"
#endif
    "  -h          print this help text\n"
    "  -i          ignore data processing\n"
    "  -r rate     frame rate of test in Hz [default 2000]\n"
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
