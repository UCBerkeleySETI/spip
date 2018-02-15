/***************************************************************************
 *
 *   Copyright (C) 2017 Andrew Jameson
 *   Licensed under the Academic Free License version 2.1
 *
 ***************************************************************************/

#include "spip/HardwareAffinity.h"
#include "spip/MeerKATPolSubXpose.h"
#ifdef HAVE_CUDA
#include "spip/MeerKATPolSubXposeCUDA.h"
#endif

#include <unistd.h>
#include <signal.h>
#include <pthread.h>

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
  string read1_key;
  string read2_key;
  string write_key;

  spip::HardwareAffinity hw_affinity;

  int verbose = 0;

#ifdef HAVE_CUDA
  int device_id = -1;
#endif

  int core = -1;

  // the sub-band that this instance should process
  int subband = -1;

  opterr = 0;
  int c;

#ifdef HAVE_CUDA
  while ((c = getopt(argc, argv, "b:d:hk:v")) != EOF) 
#else
  while ((c = getopt(argc, argv, "b:hk:v")) != EOF) 
#endif
  {
    switch(c) 
    {
      case 'b':
        core = atoi(optarg);
        break;
  
#ifdef HAVE_CUDA
      case 'd':
        device_id = atoi(optarg);
        break;
#endif

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

  // check command line arguments
  int nargs = argc - optind;
  if (nargs != 4)
  {
    cerr << "Error: 4 command line arguments are required" << endl;
    usage();
    return (EXIT_FAILURE);
  }

  if (core >= 0)
  {
    hw_affinity.bind_process_to_cpu_core (core);
    //hw_affinity.bind_to_memory (core);
  }

  read1_key = std::string(argv[optind+0]);
  read2_key = std::string(argv[optind+1]);
  write_key = std::string(argv[optind+2]);

  subband = atoi(argv[optind+3]);
  if (subband != 0 &&  subband != 1)
  {
    cerr << "Error: subband must be 0 or 1" << endl;
    usage();
    return (EXIT_FAILURE);
  }

  spip::MeerKATPolSubXpose * op;

#ifdef HAVE_CUDA
  if (device_id >= 0)
  {
    op = new spip::MeerKATPolSubXposeCUDA (read1_key.c_str(),
                                           read2_key.c_str(),
                                           write_key.c_str(),
                                           subband, device_id);
  }
  else
#endif
  {
    op = new spip::MeerKATPolSubXpose (read1_key.c_str(),
                                       read2_key.c_str(),
                                       write_key.c_str(),
                                       subband);
  }

  //if (verbose)
  cerr << "meerkat_polsubxpose: main()" << endl;
  try
  {
    op->main();
  }
  catch (std::exception& exc)
  {
    cerr << "meerkat_polsubxpose: ERROR: " << exc.what() << endl;
    return -1;
  }

  //delete op;
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
  cout << "meerkat_polsubxpose [options] in1_key in2_key out_key subband\n"
    "  in1_key     PSRDada shared memory key for pol 1\n"
    "  in2_key     PSRDada shared memory key for pol 2\n"
    "  out_key     PSRDada shared memory key for output\n"
    "  subband     sub-band to write to output[0 or 1]\n"
    "  -b core     bind computation to specified CPU core\n"
    "  -d id       GPU device on which to perform computation\n"
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
