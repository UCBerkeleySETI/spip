/***************************************************************************
 *
 *   Copyright (C) 2018 Andrew Jameson
 *   Licensed under the Academic Free License version 2.1
 *
 ***************************************************************************/

#include "config.h"

#include "spip/PBSCalibrationPipeline.h"
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

  spip::PBSCalibrationPipeline * cp;

  spip::HardwareAffinity hw_affinity;

  int verbose = 0;

  opterr = 0;
  int c;

  int core;

  // folding period in micro seconds
  double period_us = 2;

  // output sampling time
  double tsamp_out = 10;

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
        period_us = double(atof (optarg));
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

  // UWB sampling interval is fixed at 1/128 micro seconds
  double tsamp_us = 1.0f / 128.0f;

  // the number of phase bins to fold the calibration signal into
  unsigned nbin = floor(period_us / tsamp_us);

  if (tsamp_us * nbin != period_us)
  {
    cerr << "ERROR: folding period must be an integer multiple of the sampling interval" << endl;
    return EXIT_FAILURE;
  }

  // folding period in seconds
  double period_s = period_us / 1e6;

  // check limits
  if (tsamp_out < 2 * period_s)
  {
    cerr << "ERROR: output sampling interval must be at least greater than twice the folding period" << endl;
    return EXIT_FAILURE;
  }

  // number of output folds per output time sample
  unsigned folds_per_output_sample = unsigned(floor(tsamp_out / period_s));

  // the number of input time samples that will be integrated into an output time sample
  unsigned dat_dec = folds_per_output_sample * nbin;

  if (verbose)
  {
    spip::Container::verbose = true;
    cerr << "requested output sampling interval: " << tsamp_out << " seconds" << endl;
    cerr << "folds per output sample=" << folds_per_output_sample << endl;
    cerr << "input samples per output sample=" << dat_dec << endl;
  }

  cp = new spip::PBSCalibrationPipeline (in_key.c_str(), out_key.c_str());

  if (verbose)
    cp->set_verbose();

  cerr << "nbin=" << nbin << " dat_dec=" << dat_dec << endl;
  cp->set_periodicity (nbin, dat_dec);

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
  cout << "uwb_continuum_pipeline [options] inkey outkey" << endl;
  cout << " -b core    bind to CPU core" << endl;
#ifdef HAVE_CUDA
  cout << " -d gpu     use GPU" << endl;
#endif
  cout << " -c period  folding period in microseconds [default 2]" << endl;
  cout << " -t secs    desired output sampling interval in seconds [default 10]" << endl;
  cout << " -h         display usage" << endl;
  cout << " -v         verbose output" << endl;
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
