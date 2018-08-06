/***************************************************************************
 *
 *   Copyright (C) 2018 Andrew Jameson
 *   Licensed under the Academic Free License version 2.1
 *
 ***************************************************************************/

#include "config.h"

#include "spip/Error.h"
#include "spip/HardwareAffinity.h"
#include "spip/ContainerFileRead.h"
#include "spip/IntegrationRAM.h"
#include "spip/ContainerFileWrite.h"

#include <signal.h>
#include <unistd.h>

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
  // input file
  string input_file;

  // output file
  string output_file;

  spip::HardwareAffinity hw_affinity;

  int verbose = 0;

  int c;

  opterr = 0;

  int core;

  int nchan_max = -1;

  while ((c = getopt(argc, argv, "b:c:hv")) != EOF)
  {
    switch(c)
    {
      case 'b':
        core = atoi(optarg);
        hw_affinity.bind_process_to_cpu_core (core);
        hw_affinity.bind_to_memory (core);
        break;

      case 'c':
        nchan_max = atoi(optarg);
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
  int num_args = argc - optind;
  if (num_args != 2)
  {
    fprintf(stderr,"ERROR: 1 command line argument expected\n");
    usage();
    return EXIT_FAILURE;
  }

  signal(SIGINT, signal_handler);

  input_file = string(argv[optind + 0]);
  output_file = string(argv[optind + 1]);

  spip::ContainerFileRead  * input = new spip::ContainerFileRead(input_file);
  spip::ContainerFileWrite * output = new spip::ContainerFileWrite("/tmp");
  
  // create output filename
  output->set_filename (output_file);

  spip::IntegrationRAM * integrator = new spip::IntegrationRAM();
  integrator->set_input (input);
  integrator->set_output (output);
  integrator->set_verbose (verbose);

  // open the input file, reading the header
  if (verbose)
    cerr << "main: input->open_file()" << endl;
  input->open_file ();

  // read the header input parameters, determine the container size
  if (verbose)
    cerr << "main: input->read_header()" << endl;
  input->process_header();

  // determine the number of channels
  unsigned chan_dec = 1;
  if (nchan_max > 0)
  {
    unsigned nchan = input->get_nchan();
    while (int(nchan) > nchan_max)
    {
      nchan /= 2;
    }
    chan_dec = input->get_nchan() / nchan;
    if (verbose)
      cerr << "main: scrunching from " << input->get_nchan() << " to " << nchan << " [" << chan_dec << "]" << endl;
  }

  // perform a scrunch
  integrator->set_decimation (input->get_ndat(), 1, chan_dec, 1);

  // configure the transform
  if (verbose)
    cerr << "main: integrator->configure()" << endl;
  integrator->configure(input->get_order());

  // configure the output 
  if (verbose)
    cerr << "main: output->process_header()" << endl;
  output->process_header ();

  // read data from the file
  if (verbose)
    cerr << "main: input->read_data()" << endl;
  input->read_data();

  // transform the data
  if (verbose)
    cerr << "main: integrator->prepare()" << endl;
  integrator->prepare ();
  if (verbose)
    cerr << "main: integrator->transformation ()" << endl;
  integrator->transformation ();

  // write the output data
  if (verbose)
    cerr << "main: output->write()" << endl;
  output->write ();

  if (verbose)
    cerr << "main: input->close_file()" << endl;
  input->close_file ();
  if (verbose)
    cerr << "main: input->close_file()" << endl;
  output->close_file ();
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
  cout << "uwb_continuum_scrunch [options] in_file out_dir" << endl;
  cout << " in_file    input PSRDada file" << endl;
  cout << " out_file   output PSRDada file " << endl;
  cout << " -b core    bind to CPU core" << endl;
  cout << " -c nchan   fscrunch to nchan" << endl;
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
