/***************************************************************************
 *
 *   Copyright (C) 2017 Andrew Jameson
 *   Licensed under the Academic Free License version 2.1
 *
 ***************************************************************************/

#include "spip/Error.h"
#include "spip/UDPSocketSend.h"
#ifdef HAVE_VMA
#include "spip/UDPSocketReceiveVMA.h"
#else
#include "spip/UDPSocketReceive.h"
#endif
#include "spip/HardwareAffinity.h"

#include "spip/KeckRTCDefs.h"
#include "spip/stopwatch.h"

#include <unistd.h>
#include <fcntl.h>
#include <signal.h>
#include <pthread.h>

#include <cstdio>
#include <cstring>
#include <iostream>
#include <stdexcept>

#include <sys/time.h>
#include <float.h>

void usage();
void signal_handler (int signal_value);
double diff_time ( struct timeval time1, struct timeval time2 );
char quit_threads = 0;

using namespace std;

int main(int argc, char *argv[]) try
{
  string server;

  int port;

  string client;

  spip::HardwareAffinity hw_affinity;

  int verbose = 0;

  int core = -1;

  unsigned frame_size = KeckRTC_HEAP_SIZE;

  unsigned packet_size = KeckRTC_UDP_SIZE;

  unsigned frame_rate = 2000; // hertz

  unsigned duration = 10; // seconds

  int network_rate = -1;

  opterr = 0;
  int c;

  while ((c = getopt(argc, argv, "b:d:f:hn:p:r:v")) != EOF) 
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
  
      case 'h':
        cerr << "Usage: " << endl;
        usage();
        exit(EXIT_SUCCESS);
        break;

      case 'n':
        network_rate = atoi(optarg);
        break;

      case 'p':
        packet_size = atoi(optarg);
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

  signal(SIGINT, signal_handler);

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

  server = std::string(argv[optind+0]);
  client = std::string(argv[optind+2]);
  port = atoi(argv[optind+1]);

  // create a UDP sending socket
  spip::UDPSocketSend sock_send;
  if (verbose)
    cerr << "opening sending socket to " << server << ":" << port 
         << " from " << client << endl;
  sock_send.open (server, port, client);
  sock_send.resize (packet_size);

  port++;

#ifdef HAVE_VMA
  spip::UDPSocketReceiveVMA sock_recv;
#else
  spip::UDPSocketReceive sock_recv;
#endif

  if (verbose)
    cerr << "opening receiving socket on " << client << ":" << port << endl;
  sock_recv.open (client, port);
  sock_recv.resize (packet_size);
  sock_recv.set_block ();

  stopwatch_t frame_sw;
  stopwatch_t wait_sw;
  double sleep_time = 0;
  if (network_rate > 0)
  {
    //                                      Gb/s           b/s  B/us
    double bytes_per_microsecond = double(network_rate) * (1e9 / 8);
    sleep_time = packet_size / bytes_per_microsecond;
    cerr << "sleep_time=" << sleep_time *1e6 << " seconds" << endl;
  }

  struct timeval start_time;
  struct timeval start_frame;
  struct timeval end_frame;
  struct timeval curr_time;
  struct timeval end_time;

  double time_sum = 0;
  uint64_t packets_sent = 0;

  uint64_t frames_to_send = frame_rate * duration;
  uint64_t iframe = 0;

  double * times = (double *) malloc (frames_to_send * sizeof(double));

  // warm up phase
  for (iframe=0; iframe < frames_to_send; iframe++)
  {
    times[iframe] = double(iframe);
  }

  iframe = 0;

  // start of the data transfer
  gettimeofday(&start_time, 0);

  // transmit phase
  while (iframe < frames_to_send && !quit_threads)
  {
    StartTimer(&frame_sw);

    uint64_t bytes_sent = 0;
    while (bytes_sent < frame_size)
    {
      if (network_rate > 0)
        StartTimer(&wait_sw);

      // send a packet
      if (verbose)
        cerr << "Sending " << packet_size << " bytes" << endl;
      sock_send.send (packet_size);
      bytes_sent += packet_size;

      if (network_rate > 0)
        DelayTimer(&wait_sw, sleep_time);
    }

    // receive a reply
    if (verbose)
      cerr << "Receiving a reply" << endl;
    size_t reply_size = sock_recv.recv_from();
    sock_recv.consume_packet();
    StopTimer(&frame_sw);

    if (verbose)
      cerr << "Received " << reply_size << " bytes" << endl;

    unsigned long frame_time_ns = ReadTimer(&frame_sw);
    times[iframe] = double(frame_time_ns) / 1000;
    time_sum += times[iframe];

    packets_sent++;
    iframe++;

    // now get the current time to ensure the frame rate is correct
    gettimeofday (&curr_time, 0);
    double curr_offset = diff_time(start_time, curr_time);
    double next_frame_offset = iframe * (double(1e6) / frame_rate);
    while (curr_offset < next_frame_offset)
    {
      gettimeofday (&curr_time, 0);
      curr_offset = diff_time(start_time, curr_time);
    }
  }

  // start of the data transfer
  gettimeofday(&end_time, 0);

  double total_time = diff_time(start_time, end_time);

  cerr << "Sending " << packet_size-1 << " bytes" << endl;
  sock_send.send (packet_size-1);

  sock_send.close_me();
  sock_recv.close_me();

  uint64_t discard_frames = 100;

  if (iframe > 0)
  {
    uint64_t frames_sent = iframe - 1;
    uint64_t frames_counted = frames_sent - discard_frames;
  
    double duration_min = DBL_MAX;
    double duration_max = -DBL_MAX;
    double duration_sum = 0;

    int flags = O_WRONLY | O_CREAT | O_TRUNC;
    int perms = S_IRUSR | S_IWUSR | S_IRGRP | S_IROTH;
    int fd = open ("timing.dat", flags, perms);
    ::write (fd, times+discard_frames, frames_counted * sizeof(double));
    ::close (fd);

    time_sum = 0;

    for (iframe=discard_frames; iframe<frames_sent; iframe++)
    {
      double duration = times[iframe];
      
      if (duration < duration_min)
        duration_min = duration;
      if (duration > duration_max)
        duration_max = duration;
      duration_sum += duration;

      time_sum += duration;
    }


    double duration_mean = duration_sum / double(frames_counted);
    cerr << "duration timing:" << endl;
    cerr << "  minimum=" << duration_min << " us" << endl;
    cerr << "  mean="    << duration_mean << " us" << endl;
    cerr << "  maximum=" << duration_max << " us" << endl;

    double bytes_per_microsecond = (frames_counted * frame_size) / time_sum;
    double gb_per_second = (bytes_per_microsecond * 1000000) / 1000000000;

    double frames_per_microsecond = frames_sent / total_time;
    double frames_per_second = frames_per_microsecond *1e6;

    cerr << "  data_rate=" << gb_per_second << " Gb/s" << endl;
    cerr << "  frame_rate=" << frames_per_second << endl;
  }
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
  cout << "keckrtc_client [options] server port client\n"
    "  receiver    server hostname/IP address\n"
    "  port        port for send and receiver\n"
    "  client      client hostname/IP address\n"
    "  -b core     bind computation to specified CPU core\n"
    "  -d num      run test for num seconds [default 10]\n"
    "  -f bytes    size of a frame in bytes [default "<< KeckRTC_HEAP_SIZE << "]\n"
    "  -n rate     limit data transmission to rate Gb/s [default no limit]\n"
    "  -p bytes    size of a UDP packet in bytes [default " << KeckRTC_UDP_SIZE << "]\n"
    "  -r rate     run test at frame rate [default 2000Hz]\n"
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
