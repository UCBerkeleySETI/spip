/***************************************************************************
 *
 *   Copyright (C) 2017 Andrew Jameson
 *   Licensed under the Academic Free License version 2.1
 *
 ***************************************************************************/

#include "spip/Error.h"
#include "spip/UDPSocketSend.h"
#include "spip/UDPSocketReceive.h"
#include "spip/HardwareAffinity.h"

#include "spip/KeckRTCDefs.h"

#include <unistd.h>
#include <signal.h>
#include <pthread.h>

#include <cstdio>
#include <cstring>
#include <iostream>
#include <stdexcept>

#include <sys/time.h>

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

  opterr = 0;
  int c;

  while ((c = getopt(argc, argv, "b:hv")) != EOF) 
  {
    switch(c) 
    {
      case 'b':
        core = atoi(optarg);
        break;
  
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

  // UDP packet size for send/recv
  size_t bufsz = KeckRTC_UDP_SIZE;

  // create a UDP sending socket
  spip::UDPSocketSend sock_send;
  if (verbose)
    cerr << "opening sending socket to " << server << ":" << port 
         << " from " << client << endl;
  sock_send.open (server, port, client);
  sock_send.resize (bufsz);

  port++;

  spip::UDPSocketReceive sock_recv;
  if (verbose)
    cerr << "opening receiving socket on " << client << ":" << port << endl;
  sock_recv.open (client, port);
  sock_recv.resize (bufsz);
  sock_recv.set_block ();

  struct timeval start_time;
  struct timeval end_time;
  double time_taken = 0;
  double time_sum = 0;
  uint64_t packets_sent = 0;

  while (!quit_threads)
  {
    gettimeofday (&start_time, 0);

    uint64_t bytes_sent = 0;
    while (bytes_sent < KeckRTC_HEAP_SIZE)
    {
      // send a packet
      if (verbose)
        cerr << "Sending " << bufsz << " bytes" << endl;
      sock_send.send (bufsz);
      bytes_sent += bufsz;
    }

    // receive a reply
    if (verbose)
      cerr << "Receiving a reply" << endl;
    size_t reply_size = sock_recv.recv_from();
    sock_recv.consume_packet();

    gettimeofday (&end_time, 0);
    if (verbose)
      cerr << "Received " << reply_size << " bytes" << endl;

    time_taken = diff_time (start_time, end_time);

    time_sum += time_taken;
    packets_sent++;

    if (packets_sent % 10000 == 0)
    {
      time_taken = time_sum / double(packets_sent);
      fprintf (stderr, "Time %10.6lf microseconds\n", time_taken);
      packets_sent = 0;
      time_sum = 0;
    }
  }

  cerr << "Sending " << bufsz-1 << " bytes" << endl;
  sock_send.send (bufsz-1);

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
  cout << "keckrtc_client [options] server port client\n"
    "  receiver    server hostname/IP address\n"
    "  port        port for send and receiver\n"
    "  client      client hostname/IP address\n"
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

double diff_time ( struct timeval time1, struct timeval time2 )
{
  return ( double(time2.tv_sec - time1.tv_sec) * 1000000 + 
           double(time2.tv_usec - time1.tv_usec ) );
}

