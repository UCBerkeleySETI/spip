/***************************************************************************
 *
 *   Copyright (C) 2016 Andrew Jameson
 *   Licensed under the Academic Free License version 2.1
 *
 ***************************************************************************/

#include "config.h"

#include "spip/ContinuumPipeline.h"
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

  spip::ContinuumPipeline * dp;

  spip::HardwareAffinity hw_affinity;

  int verbose = 0;

  opterr = 0;
  int c;

  int core;

  int nchan = 1024;

  int channel_oversampling = 32;

  float tsamp_out = 10;

  int output_npol = 1;

#ifdef HAVE_CUDA
  int device = -1;

  while ((c = getopt(argc, argv, "b:d:hn:o:p:t:v")) != EOF)
#else
  while ((c = getopt(argc, argv, "b:hn:o:p:t:v")) != EOF)
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

      case 'n':
        nchan = atoi (optarg);
        break;

      case 'o':
        channel_oversampling = atoi (optarg);
        break;

      case 'p':
        output_npol = atoi(optarg);
        break;

      case 't':
        tsamp_out = atof (optarg);
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

  // input time sampling is fixed at 1/128 micro seconds
  double tsamp = 1.0f / (128.0f * 1e6);
  double tsamp_channelised = tsamp * nchan * channel_oversampling;

  cerr << "tsamp=" << tsamp << " tsamp_channelised=" << tsamp_channelised << endl;

  // check if the requested sampling interval is greater than the channelised one
  if (tsamp_out < tsamp_channelised)
  {
    fprintf (stderr, "ERROR: requested output sampling interval [%le s], is lessthan the minimum [%le s]\n", 
             tsamp_out, tsamp_channelised);
    return EXIT_FAILURE;
  }

  int tdec_out = int(floor(tsamp_out/tsamp_channelised));
  double tsamp_actual = tsamp_channelised * tdec_out;
  cout << "Requested sampling time: " << tsamp_out << "s actual: " << tsamp_actual << "s TDEC=" << tdec_out << endl;

  dp = new spip::ContinuumPipeline (in_key.c_str(), out_key.c_str());

  if (verbose)
    dp->set_verbose();

  cerr << "Channelisation: " << nchan<< endl;
  dp->set_channelisation (nchan);
  dp->set_channel_oversampling (channel_oversampling);
  dp->set_decimation (tdec_out);

  if (output_npol == 1)
    dp->set_output_state (spip::Signal::Intensity);
  else if (output_npol == 2)
    dp->set_output_state (spip::Signal::PPQQ);
  else if (output_npol == 4)
    dp->set_output_state (spip::Signal::Coherence);
  else
    throw invalid_argument("unrecognized output polarisation state");

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
  cout << "uwb_continuum_pipeline [options] inkey outkey" << endl;
  cout << " -b core    bind to CPU core" << endl;
#ifdef HAVE_CUDA
  cout << " -d gpu     use GPU" << endl;
#endif
  cout << " -n nchan   number of output channels [default 1024]" << endl;
  cout << " -o factor  channel oversampling factor [default 32]" << endl;
  cout << " -p npol    number of output polarisations, 1: Intensity, 2: PPQQ, 4: Coherence [default 1]" << endl;
  cout << " -t secs    desired output sampling interval [default 10]" << endl;
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
