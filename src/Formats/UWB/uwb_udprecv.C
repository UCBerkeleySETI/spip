/***************************************************************************
 *
 *    Copyright (C) 2015 by Andrew Jameson
 *    Licensed under the Academic Free License version 2.1
 *
 ****************************************************************************/

#include "config.h"

#include "spip/AsciiHeader.h"
#include "spip/UDPReceiver.h"
#include "spip/UDPFormatVDIF.h"
#include "spip/UDPFormatDualVDIF.h"
#include "spip/UDPFormatUWB.h"
#include "spip/Error.h"

#ifdef HAVE_HWLOC
#include "spip/HardwareAffinity.h"
#endif

#include <unistd.h>
#include <signal.h>
#include <pthread.h>

#include <cstdio>
#include <cstring>
#include <iostream>

void usage();
void * stats_thread (void * arg);
void signal_handler (int signal_value);
char quit_threads = 0;

spip::UDPReceiver * udprecv;

using namespace std;

int main(int argc, char *argv[])
{
  spip::AsciiHeader config;

#ifdef HAVE_HWLOC
  // core on which to bind thread operations
  spip::HardwareAffinity hw_affinity;
#endif

  string * format_name = new string("vdif");

  int core = -1;

  char verbose = 0;

  opterr = 0;
  int c;

  while ((c = getopt(argc, argv, "b:f:hv")) != EOF) 
  {
    switch(c) 
    {
      case 'b':
        core = atoi(optarg);
#ifdef HAVE_HWLOC
        hw_affinity.bind_process_to_cpu_core (core);
        hw_affinity.bind_to_memory (core);
#endif
        break;

      case 'f':
        format_name = new string(optarg);
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

  // Check arguments
  if ((argc - optind) != 1) 
  {
    fprintf(stderr,"ERROR: 1 command line argument expected\n");
    usage();
    return EXIT_FAILURE;
  }

  signal(SIGINT, signal_handler);

  try 
  {
    // create a UDP Receiver
    udprecv = new spip::UDPReceiver();
    udprecv->verbose = verbose;

    spip::UDPFormatVDIF * format;
    if (format_name->compare("uwb") == 0)
      format = new spip::UDPFormatUWB();
    else if (format_name->compare("vdif") == 0)
      format = new spip::UDPFormatVDIF(); 
    else if (format_name->compare("dualvdif") == 0)
      format = new spip::UDPFormatDualVDIF(); 
    else
    {
      cerr << "ERROR: unrecognized UDP format [" << format << "]" << endl;
      delete udprecv;
      return (EXIT_FAILURE);
    } 
    format->set_self_start (true);
    udprecv->set_format (format);

    if (verbose)
      cerr << "uwb_udprecv: Loading configuration from " << argv[optind] << endl;

    // config file for this data stream
    config.load_from_file (argv[optind]);

    if (udprecv->verbose)
      cerr << "uwb_udprecv: configuring using fixed config" << endl;
    udprecv->configure (config.raw());

    if (udprecv->verbose)
      cerr << "uwb_udprecv: allocating runtime resources" << endl;
    udprecv->prepare ();

    if (udprecv->verbose)
      cerr << "uwb_udprecv: starting stats thread" << endl;
    pthread_t stats_thread_id;
    int rval = pthread_create (&stats_thread_id, 0, stats_thread, (void *) udprecv);
    if (rval != 0)
    {
      cerr << "uwb_udprecv: failed to start stats thread" << endl;
      return (EXIT_FAILURE);
    }

    if (udprecv->verbose)
      cerr << "uwb_udprecv: receiving" << endl;
    udprecv->receive ();

    quit_threads = 1;

    if (udprecv->verbose)
      cerr << "uwb_udprecv: joining stats_thread" << endl;
    void * result;
    pthread_join (stats_thread_id, &result);
  
    delete udprecv;
  }
  catch (Error& error)
  {
    cerr << error << endl;
    exit (-1);
  }
  catch (std::exception& exc)
  {
    cerr << "uwb_udprecv: ERROR: " << exc.what() << endl;
    return -1;
  }

  return 0;
}

void usage() 
{
  cout << "uwb_udprecv [options] config\n"
    "  header      ascii file contain config and header\n"
    "  -b core     bind computation to specified CPU core\n"
    "  -f format   UDP format to use: vdif, dualvdif or uwb [default vdif]\n"
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
  udprecv->stop_receiving();
}


/* 
 *  Thread to print simple capture statistics
 */
void * stats_thread (void * arg)
{
  spip::UDPReceiver * recv = (spip::UDPReceiver *) arg;

  uint64_t b_recv_total = 0;
  uint64_t b_recv_curr = 0;
  uint64_t b_recv_1sec;

  uint64_t s_curr = 0;
  uint64_t s_total = 0;
  uint64_t s_1sec;

  uint64_t b_drop_curr = 0;

  float gb_recv_ps = 0;
  float mb_recv_ps = 0;

  while (!quit_threads)
  {
    // get a snapshot of the data as quickly as possible
    b_recv_curr = recv->get_stats()->get_data_transmitted();
    b_drop_curr = recv->get_stats()->get_data_dropped();
    s_curr      = recv->get_stats()->get_nsleeps();

    // calc the values for the last second
    b_recv_1sec = b_recv_curr - b_recv_total;
    s_1sec = s_curr - s_total;

    // update the totals
    b_recv_total = b_recv_curr;
    s_total = s_curr;

    mb_recv_ps = (double) b_recv_1sec / 1000000;
    gb_recv_ps = (mb_recv_ps * 8)/1000;

    // determine how much memory is free in the receivers
    fprintf (stderr,"Recv %6.3f [Gb/s] Sleeps %lu Dropped %lu B\n", gb_recv_ps, s_1sec, b_drop_curr);
    sleep(1);
  }
 
  cerr << "stats_thread: exiting" << endl;
  return (void *) 0;
}

