/***************************************************************************
 *
 *   Copyright (C) 2016 Andrew Jameson
 *   Licensed under the Academic Free License version 2.1
 *
 ***************************************************************************/

#include "config.h"

#include "spip/PreprocessingPipeline.h"
#include "spip/HardwareAffinity.h"
#include "spip/UnpackFloatRAMUWB.h"

#if HAVE_CUDA
#include "spip/UnpackFloatCUDAUWB.h"
#endif

#include <signal.h>

#include <cmath>
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

  string cal_path = ".";

  string trans_key;

  string out_key;

  spip::PreprocessingPipeline * pp;

  spip::HardwareAffinity hw_affinity;

  int verbose = 0;

  opterr = 0;
  int c;

  int core;

  int nfft = 1024;

  float cal_tsamp_out = 10; // in seconds

  float cal_freq_res = 1;   // in MHz

  float trans_tsamp_out = 64; // in microseconds

  bool filter = false;

  bool calibrate = false;

  bool transients = false;

  int ref_pol = -1;

  double req_mon_tsamp = 10;

#ifdef HAVE_CUDA
  int device = -1;

  while ((c = getopt(argc, argv, "abc:d:e:f:hn:r:t:v")) != EOF)
#else
  while ((c = getopt(argc, argv, "abc:e:f:hn:r:t:v")) != EOF)
#endif
  {
    switch(c)
    {
      case 'a':
        filter = true;
        break;

      case 'b':
        core = atoi(optarg);
        hw_affinity.bind_process_to_cpu_core (core);
        hw_affinity.bind_to_memory (core);
        break;

      case 'c':
        calibrate = true;
        cal_tsamp_out = atof (optarg);
        break;

#ifdef HAVE_CUDA
      case 'd':
        device = atoi(optarg);
        break;
#endif

      case 'e':
        calibrate = true;
        cal_freq_res = atof (optarg);
        break;

      case 'f':
        transients = true;
        trans_tsamp_out = atof (optarg);
        break;

      case 'h':
        cerr << "Usage: " << endl;
        usage();
        exit(EXIT_SUCCESS);
        break;

      case 'n':
        nfft = atoi (optarg);
        break;

      case 'r':
        ref_pol = atoi(optarg);
        break;

      case 't':
        req_mon_tsamp = double(atof(optarg));
        break;

      case 'v':
        verbose++;
        spip::Container::verbose = true;
        break;

      default:
        cerr << "Unrecognised option [" << char(c) << "]" << endl;
        usage();
        return EXIT_FAILURE;
        break;
    }
  }

  // Check arguments
  if ((argc - optind) != 3)
  {
    fprintf(stderr,"ERROR: 3 command line arguments expected\n");
    usage();
    return EXIT_FAILURE;
  }

  signal(SIGINT, signal_handler);

  in_key = argv[optind];
  trans_key = argv[optind+1];
  out_key = argv[optind+2];

  pp = new spip::PreprocessingPipeline (in_key.c_str(), cal_path.c_str(), trans_key.c_str(), out_key.c_str());

  if (verbose)
    pp->set_verbose();

  cerr << "main: calibrate=" << calibrate << " filter=" << filter << " transients=" << transients << endl;
  pp->set_function (calibrate, filter, transients);

  cerr << "main: ref_pol=" << ref_pol << " mon_tsamp=" << req_mon_tsamp << endl;
  pp->set_filtering (ref_pol, req_mon_tsamp);

  cerr << "main: channelisation=" << nfft << endl;
  pp->set_channelisation (nfft);

  double tsamp = 1.0f / (128.0f);

  double tsamp_channelised, bw, cal_bw;
  int tdec_out;
  tsamp = 1.0f / (128.0f);

  tsamp_channelised = tsamp * nfft;
  tdec_out = int(floor(trans_tsamp_out/tsamp_channelised));
  cerr << "main: Trans tdec=" << tdec_out << " pdec=2" << endl;
  pp->set_trans_decimation (tdec_out, 2);

  tsamp = 1.0f / (128.0f * 1e6);
  tsamp_channelised = tsamp * nfft;
  tdec_out = int(floor(cal_tsamp_out/tsamp_channelised));

  bw = 128.0;   // MHz
  cal_bw = cal_freq_res; // MHz
  int fdec_out = int(floor(cal_bw / (bw/nfft)));
  cerr << "main: Cal tdec=" << tdec_out << " pdec=1" << endl;
  pp->set_cal_decimation (fdec_out, tdec_out, 1);

#ifdef HAVE_CUDA
  if (device >= 0)
  {
    pp->set_device (device);
    pp->configure (new spip::UnpackFloatCUDAUWB());
  }
  else
  #endif
  {
    pp->configure (new spip::UnpackFloatRAMUWB());
  }
  pp->open ();
  pp->process ();
  pp->close ();

  delete pp;
}
catch (Error& error)
{
  cerr << error << endl;
  exit (-1);
}
catch (std::exception& exc)
{
  cerr << "ERROR: " << exc.what() << endl;
  exit (-1);
}

void usage()
{
  cout << "uwb_preprocessing_pipeline [options] inkey transkey outkey" << endl;
  cout << " -a        apply adaptive filtering" << endl;
  cout << " -c tsamp  write binned calibration spectra with tsamp in seconds [default 10]" << endl;
#ifdef HAVE_CUDA
  cout << " -d gpu    use GPU [default: use CPU]" << endl;
#endif
  cout << " -e fres   write binned calibration spectra to calkey with frequnecy resolution in MHz [default 1]" << endl;
  cout << " -f tsamp  write transient search mode filterbanks to transkey with specificed sampling interface in microseconds [default 64]" << endl;
  cout << " -h        display usage" << endl;
  cout << " -n nfft   FFT length for filtering [default: 1024]" << endl;
  cout << " -r ipol   polarisatsion ipol contains an RFI reference signal" << endl;
  cout << " -t tsamp  Adaptive filter monitoring tsamp in seconds [default 10]" << endl;
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
