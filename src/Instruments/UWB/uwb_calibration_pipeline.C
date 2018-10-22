/***************************************************************************
 *
 *   Copyright (C) 2018 Andrew Jameson
 *   Licensed under the Academic Free License version 2.1
 *
 ***************************************************************************/

#include "config.h"

#include "spip/CalibrationPipeline.h"
#include "spip/UnpackFloatRAMUWB.h"
#include "spip/HardwareAffinity.h"

#if HAVE_CUDA
#include "spip/UnpackFloatCUDAUWB.h"
#endif

#include <signal.h>
#include <math.h>

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

  spip::CalibrationPipeline * cp;

  spip::HardwareAffinity hw_affinity;

  int verbose = 0;

  opterr = 0;
  int c;

  int core;

  // number of output channels 
  unsigned nfft = 128;

  // output sampling time
  double tsamp_out = 10;

  // perform polarisation scrunching
  bool pscrunch = false;

#ifdef HAVE_CUDA
  int device = -1;

  while ((c = getopt(argc, argv, "b:c:d:hn:p:t:v")) != EOF)
#else
  while ((c = getopt(argc, argv, "b:c:hn:p:t:v")) != EOF)
#endif
  {
    switch(c)
    {
      case 'b':
        core = atoi(optarg);
        hw_affinity.bind_process_to_cpu_core (core);
        hw_affinity.bind_to_memory (core);
        break;

      case 'c':
        nfft = atoi (optarg);
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

      case 'p':
        pscrunch = true;
        break;

      case 't':
        tsamp_out = double(atof (optarg));
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

  unsigned channel_oversampling = 8;

  // UWB sampling interval is fixed at 1/128 micro seconds
  double tsamp_in = 1.0f / 128000000.0f;

  double tsamp_channelised = tsamp_in * nfft * channel_oversampling;
  
  // averaging time for TSYS measurement is 1s [for now]
  uint64_t dat_dec = uint64_t (tsamp_out / tsamp_channelised);
  unsigned pol_dec = 1;
  if (pscrunch)
    pol_dec = 2;

  cerr << "uwb_calibration_pipeline: dat_dec=" << dat_dec << " pol_dec=" << pol_dec << " nfft=" << nfft << endl;

  cp = new spip::CalibrationPipeline (in_key.c_str(), out_key.c_str());

  if (verbose)
    cp->set_verbose();

  cp->set_channelisation (nfft * channel_oversampling);
  cp->set_decimation (dat_dec, pol_dec, channel_oversampling);

  // TODO think about signal state
  //cp->set_output_state (spip::Signal::Coherence);

#ifdef HAVE_CUDA
  if (device >= 0)
  {
    cp->set_device (device);
    cp->configure (new spip::UnpackFloatCUDAUWB());
  }
  else
#endif
  {
    cp->configure (new spip::UnpackFloatRAMUWB());
  }
  
  cp->open ();
  cp->process ();
  cp->close ();

  delete cp;
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
  cout << "uwb_calibration_pipeline [options] inkey outkey" << endl;
  cout << " -b core      bind to CPU core" << endl;
  cout << " -c channels  channelisation [default 128]" << endl;
#ifdef HAVE_CUDA
  cout << " -d gpu       use GPU" << endl;
#endif
  cout << " -p           scrunch polarisations to produce Stokes I" << endl;
  cout << " -t secs      desired output sampling interval in seconds [default 10]" << endl;
  cout << " -h           display usage" << endl;
  cout << " -v           verbose output" << endl;
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
