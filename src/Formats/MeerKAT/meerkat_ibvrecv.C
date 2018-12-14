/***************************************************************************
 *
 *    Copyright (C) 2015 by Andrew Jameson
 *    Licensed under the Academic Free License version 2.1
 *
 ****************************************************************************/


#include "spip/AsciiHeader.h"
#include "spip/HardwareAffinity.h"
#include "spip/IBVReceiver.h"
#include "spip/UDPFormatMeerKATSimple.h"
#include "spip/UDPFormatMeerKATSPEAD.h"
#include "spip/UDPFormatMeerKATSPEAD1k.h"

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

spip::IBVReceiver * ibvrecv;

using namespace std;

int main(int argc, char *argv[])
{
  string * format = new string("simple");

  spip::AsciiHeader config;

  // core on which to bind thread operations
  spip::HardwareAffinity hw_affinity;

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
        hw_affinity.bind_process_to_cpu_core (core);
        hw_affinity.bind_to_memory (core);
        break;

      case 'f':
        format = new string(optarg);
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
    ibvrecv = new spip::IBVReceiver();
    ibvrecv->verbose = verbose;

    if (verbose)
      cerr << "meerkat_ibvrecv: configuring format to be " << format << endl;

    if (format->compare("simple") == 0)
      ibvrecv->set_format (new spip::UDPFormatMeerKATSimple());
#ifdef HAVE_SPEAD2
    else if (format->compare("spead") == 0)
    {
      ibvrecv->set_format (new spip::UDPFormatMeerKATSPEAD());
    }
    else if (format->compare("spead1k") == 0)
    {
      ibvrecv->set_format (new spip::UDPFormatMeerKATSPEAD1k());
    }
#endif
    else
    {
      cerr << "ERROR: unrecognized UDP format [" << format << "]" << endl;
      delete ibvrecv;
      return (EXIT_FAILURE);
    }

    if (verbose)
      cerr << "meerkat_ibvrecv: Loading configuration from " << argv[optind] << endl;

    // config file for this data stream
    config.load_from_file (argv[optind]);

    if (ibvrecv->verbose)
      cerr << "meerkat_ibvrecv: configuring using fixed config" << endl;
    ibvrecv->configure (config.raw());

    if (ibvrecv->verbose)
      cerr << "meerkat_ibvrecv: allocating runtime resources" << endl;
    ibvrecv->prepare ();

    if (ibvrecv->verbose)
      cerr << "meerkat_ibvrecv: starting stats thread" << endl;
    pthread_t stats_thread_id;
    int rval = pthread_create (&stats_thread_id, 0, stats_thread, (void *) ibvrecv);
    if (rval != 0)
    {
      cerr << "meerkat_ibvrecv: failed to start stats thread" << endl;
      return (EXIT_FAILURE);
    }

    if (ibvrecv->verbose)
      cerr << "meerkat_ibvrecv: receiving" << endl;
    ibvrecv->receive ();

    quit_threads = 1;

    if (ibvrecv->verbose)
      cerr << "meerkat_ibvrecv: joining stats_thread" << endl;
    void * result;
    pthread_join (stats_thread_id, &result);
  
    delete ibvrecv;
  }
  catch (std::exception& exc)
  {
    cerr << "meerkat_ibvrecv: ERROR: " << exc.what() << endl;
    return -1;
  }
  cerr << "meerkat_ibvrecv: exiting" << endl;

  return 0;
}

void usage() 
{
  cout << "meerkat_ibvrecv [options] config\n"
    "  header      ascii file contain config and header\n"
#ifdef HAVE_SPEAD2
    "  -f format   receive UDP data of format [simple spead]\n"
#else
    "  -f format   receive UDP data of format [simple]\n"
#endif
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
  ibvrecv->stop_receiving();
}


/* 
 *  Thread to print simple capture statistics
 */
void * stats_thread (void * arg)
{
  uint64_t b_recv_total = 0;
  uint64_t b_recv_curr = 0;
  uint64_t b_recv_1sec;
  uint64_t b_drop_total = 0;
  uint64_t b_drop_curr = 0;
  uint64_t b_drop_1sec;

  uint64_t s_curr = 0;
  uint64_t s_total = 0;
  uint64_t s_1sec;

  float gb_recv_ps = 0;
  float mb_recv_ps = 0;
  float gb_drop_ps = 0;
  float mb_drop_ps = 0;

  while (!quit_threads)
  {
    if (!quit_threads)
    {
      // get a snapshot of the data as quickly as possible
      b_recv_curr = ibvrecv->get_stats()->get_data_transmitted();
      b_drop_curr = ibvrecv->get_stats()->get_data_dropped();
      s_curr = ibvrecv->get_stats()->get_nsleeps();

      // calc the values for the last second
      b_recv_1sec = b_recv_curr - b_recv_total;
      b_drop_1sec = b_drop_curr - b_drop_total;
      s_1sec = s_curr - s_total;

      // update the totals
      b_recv_total = b_recv_curr;
      b_drop_total = b_drop_curr;
      s_total = s_curr;

      mb_recv_ps = (double) b_recv_1sec / 1000000;
      gb_recv_ps = (mb_recv_ps * 8)/1000;
      mb_drop_ps = (double) b_drop_1sec / 1000000;
      gb_drop_ps = (mb_drop_ps * 8)/1000;

      // determine how much memory is free in the receivers
      fprintf (stderr,"Recv %6.3f [Gb/s] Drop %6.3f [Gb/s] Sleeps %lu Dropped %lu B\n", gb_recv_ps, gb_drop_ps, s_1sec, b_drop_curr);
      sleep(1);
    }
  }
  void * result = NULL;
  return result;
}

