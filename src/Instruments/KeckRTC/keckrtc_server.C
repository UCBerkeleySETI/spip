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
#ifdef HAVE_GDR
#include "gdrapi.h"
#endif
#endif
#include "spip/KeckRTCDefs.h"
#include "spip/KeckRTCUtil.h"
#include "spip/stopwatch.h"

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

#ifdef HAVE_GDR
  bool use_gdr = false;
#endif
#endif

  int core = -1;

  opterr = 0;
  int c;

#ifdef HAVE_CUDA
#ifdef HAVE_GDR
  while ((c = getopt(argc, argv, "b:d:f:g:hijr:v")) != EOF) 
#else
  while ((c = getopt(argc, argv, "b:d:f:g:hir:v")) != EOF) 
#endif // ! HAVE_GDR
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

      case 'f':
        frame_size = atoi(optarg);
        break;

#ifdef HAVE_CUDA
      case 'g':
        device_id = atoi(optarg);
        break;

#ifdef HAVE_GDR
      case 'j':
	use_gdr = true;
        break;
#endif
#endif

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
    //hw_affinity.bind_process_to_cpu_core (core);
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
  spip::UDPSocketReceiveVMA * sock_recv = new spip::UDPSocketReceiveVMA();
#else
  spip::UDPSocketReceive * sock_recv = new spip::UDPSocketReceive();
#endif
  cerr << "opening recv socket on " << server << ":" << port << endl;
  sock_recv->open (server, port);
  sock_recv->resize (bufsz + 64);
  sock_recv->set_block ();

  port++;

  spip::UDPSocketSend * sock_send = new spip::UDPSocketSend();
  cerr << "opening send socket to " << client << ":" << port << " from  " << server << endl;
  sock_send->open (client, port, server);
  sock_send->resize (bufsz);

  void * send_buf_ptr = (void *) sock_send->get_buf();
  void * recv_buf_ptr = (void *) sock_recv->get_buf();
  void * host_buf;

#ifdef HAVE_CUDA
  unsigned int flags = 0;
  cudaError_t rval;

  // register the udp socket buffers as host memory
  rval = cudaHostRegister (send_buf_ptr, bufsz, flags);
  if (rval != cudaSuccess)
    cerr << "cudaHostRegister failed on sock_send" << endl;

  // not sure I can do this with a VMA socket
  rval = cudaHostRegister (recv_buf_ptr, bufsz, flags);
  if (rval != cudaSuccess)
    cerr << "cudaHostRegister failed on sock_recv" << endl;

  // allocat host memory
  void * dev_buf;
  rval = cudaMallocHost (&host_buf, frame_size);
  if (rval != cudaSuccess)
    cerr << "cudaMallocHost (&host_buf, " << frame_size << ")" << endl;

#ifdef HAVE_GDR
  CUdeviceptr d_A;
  unsigned int flag = 1;
  gdr_t g;
  gdr_mh_t mh;
  gdr_info_t info;

  void * bar_ptr  = NULL;
  int off = 0;
  uint32_t * buf_ptr = NULL;

  if (use_gdr)
  {
    // allocate device memory
    cuMemAlloc(&d_A, frame_size);
    cuPointerSetAttribute (&flag, CU_POINTER_ATTRIBUTE_SYNC_MEMOPS, d_A);

    // open the GDR context
    g = gdr_open();

    // pin the buffer
    gdr_pin_buffer (g, d_A, frame_size, 0, 0, &mh);

    // pointer to the bar
    gdr_map(g, mh, &bar_ptr, frame_size);

    // determine any offset in the virtual address
    gdr_get_info(g, mh, &info);
    off = info.va - d_A;

    // virtual address pointer
    buf_ptr = (uint32_t *)((char *) bar_ptr + off);

    // gpu device pointer
    dev_buf = (void *) (uintptr_t) d_A;
  }
  else
#endif
  // NO GDR
  {
    // allocate device memory
    rval = cudaMalloc (&dev_buf, frame_size);
    if (rval != cudaSuccess)
      cerr << "cudaMalloc failed on dev_buf" << endl;
  }
#else
  // no CUDA/GDR
  host_buf = malloc (frame_size);
#endif
  char * host_ptr = (char *) host_buf;

  stopwatch_t frame_sw;

  std::vector<double> times;
  uint64_t frames_to_receive = frame_rate * duration;
  times.resize(frames_to_receive + 1);

  uint64_t iframe = 0;

  cerr << "ready to receive" << endl;

  // main transfer loop
  bool keep_receiving = true;
  while (keep_receiving && iframe < frames_to_receive)
  {
    // receive a whole frame, consisting multiple packets
    uint64_t bytes_received = 0;
    while (keep_receiving && bytes_received < frame_size)
    {
      // receive a packet
      size_t reply_size = sock_recv->recv_from ();
      if (reply_size == KeckRTC_UDP_SIZE)
      {
        if (process_data) 
	{
#ifdef HAVE_CUDA
#ifdef HAVE_GDR
	  if (use_gdr)
          {
            gdr_copy_to_bar (buf_ptr + bytes_received/4, sock_recv->buf_ptr, reply_size);
          }
	  else
#endif
#endif
	  {
            memcpy (host_ptr + bytes_received, sock_recv->buf_ptr, reply_size);
          }
        }
      	bytes_received += reply_size;
      }
      else
        keep_receiving = false;

      sock_recv->consume_packet();
    }

    StartTimer(&frame_sw);

#ifdef HAVE_CUDA
    if (process_data) 
    {
#ifdef HAVE_GDR
      // GDR path already has copied the data to GPU
      if (!use_gdr)
#endif
      {
        rval = cudaMemcpyAsync (dev_buf, host_ptr, frame_size, cudaMemcpyHostToDevice, stream);
        if (rval != cudaSuccess)
          cerr << "cudaMemcpyAsync failed on sock_recv" << endl;
      }

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

    StopTimer(&frame_sw);

    unsigned long frame_time_ns = ReadTimer(&frame_sw);

    // determine the time taken to perform GPU H2D, kernel and D2H
    double frame_time = double(frame_time_ns) / 1000;
    times[iframe] = frame_time;
    iframe++;

    if (keep_receiving)
    {
      if (verbose)
        cerr << "Sending reply of " << bufsz << " bytes" << endl;
      // send a reply
      sock_send->send (bufsz);
    }
  }

  if (iframe > 0)
  { 
    uint64_t frames_sent = iframe - 1;
    write_timing_data ("recv_timing.dat", times, frames_sent);
    print_timing_data (times, frames_sent, frame_size);
  }

  sock_send->close_me();
  sock_recv->close_me();

  delete sock_send;
  delete sock_recv;

#ifdef HAVE_CUDA
#ifdef HAVE_GDR
  if (use_gdr)
  {
    gdr_unmap(g, mh, bar_ptr, frame_size);
    gdr_unpin_buffer(g, mh);
    gdr_close(g);
    cuMemFree(d_A);
  }
  else
#endif
  {
    cudaFree (dev_buf);
  }

  rval = cudaHostUnregister (send_buf_ptr);
  if (rval != cudaSuccess)
    cerr << "cudaHostUnregister failed on sock_send" << endl;

  rval = cudaHostUnregister (recv_buf_ptr);
  if (rval != cudaSuccess)
    cerr << "cudaHostUnregister failed on sock_recv" << endl;

  cudaFreeHost (host_buf);
  cudaStreamDestroy(stream);

#else
  free (host_buf);
#endif

  cerr << "end main" << endl;
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
#ifdef HAVE_GDR
    "  -j          use NVIDIA GDR Copy\n"
#endif
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
