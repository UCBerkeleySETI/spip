/***************************************************************************
 *
 *   Copyright (C) 2016 Andrew Jameson
 *   Licensed under the Academic Free License version 2.1
 *
 ***************************************************************************/

#include "config.h"

#include "spip/AdaptiveFilterTest.h"
#include "spip/HardwareAffinity.h"
#include "spip/UnpackFloatRAMUWB.h"

#if HAVE_CUDA
#include "spip/UnpackFloatCUDAUWB.h"
#endif

#include <signal.h>

#include <cstdio>
#include <cstring>
#include <iostream>
#include <stdexcept>

using namespace std;

void usage();
void signal_handler (int signal_value);

char quit_threads = 0;

int main(int argc, char *argv[]) try
{
  string in_key;

  string out_key;

  spip::AdaptiveFilterTest * dp;

  spip::HardwareAffinity hw_affinity;

  int verbose = 0;

  int ref_pol = 0;

  opterr = 0;
  int c;

  int core;

#ifdef HAVE_CUDA
  int device = -1;

  while ((c = getopt(argc, argv, "b:d:hr:v")) != EOF)
#else
  while ((c = getopt(argc, argv, "b:hr:v")) != EOF)
#endif
  {
    switch(c)
    {
      case 'b':
        core = atoi(optarg);
        hw_affinity.bind_process_to_cpu_core (core);
        hw_affinity.bind_to_memory (core);
        break;

#ifdef HAVE_CUDA
      case 'd':
        device = atoi(optarg);
        break;
#endif

      case 'h':
        cerr << "Usage: " << endl;
        usage();
        exit(EXIT_SUCCESS);
        break;

      case 'r':
        ref_pol = atoi(optarg);
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
  if ((argc - optind) != 2)
  {
    fprintf(stderr,"ERROR: 2 command line argument expected\n");
    usage();
    return EXIT_FAILURE;
  }

  signal(SIGINT, signal_handler);

  in_key = argv[optind];
  out_key = argv[optind+1];

  dp = new spip::AdaptiveFilterTest (in_key.c_str(), out_key.c_str());

  if (verbose)
    dp->set_verbose();

  dp->set_filtering(ref_pol);

#ifdef HAVE_CUDA
  if (device >= 0)
  {
    dp->set_device (device);
    dp->configure_cuda (new spip::UnpackFloatCUDAUWB());
  }
  else
#endif
  {
    dp->configure (new spip::UnpackFloatRAMUWB());
  }
  dp->open ();
  dp->process ();
  dp->close ();

  delete dp;
}
catch (Error& error)
{
  cerr << error << endl;
  exit (-1);
}
catch (std::exception& exc)
{
  cerr << "ERROR: " << exc.what() << endl;
  return -1;
  return 0;
}

void usage()
{
  cout << "uwb_adaptive_filter_test [options] inkey outkey" << endl;
#ifdef HAVE_CUDA
  cout << " -d gpu    use GPU" << endl;
#endif
  cout << " -h        display usage" << endl;
  cout << " -r ipol   polarisation containing RFI reference signal [default 0]" << endl;
  cout << " -v        verbose output" << endl;
}

void signal_handler (int signalValue)
{
  fprintf(stderr, "received signal %d\n", signalValue);
  if (quit_threads)
  {
    fprintf(stderr, "received signal %d twice, hard exit\n", signalValue);
    exit(EXIT_FAILURE);
  }
  quit_threads = 1;
}
