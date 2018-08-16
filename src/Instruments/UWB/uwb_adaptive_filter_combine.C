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
#include "spip/PolCombineRAM.h"
#include "spip/AdaptiveFilterRAM.h"
#include "spip/ContainerRAM.h"
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
  // input files to add in time 
  vector<string> input_files;

  vector<spip::ContainerFileRead *> inputs;

  // output file
  string output_file;

  spip::HardwareAffinity hw_affinity;

  int verbose = 0;

  int c;

  opterr = 0;

  int core;

  while ((c = getopt(argc, argv, "b:hv")) != EOF)
  {
    switch(c)
    {
      case 'b':
        core = atoi(optarg);
        hw_affinity.bind_process_to_cpu_core (core);
        hw_affinity.bind_to_memory (core);
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
  if (num_args < 3)
  {
    fprintf(stderr,"ERROR: at least 3 command line argument expected\n");
    usage();
    return EXIT_FAILURE;
  }

  signal(SIGINT, signal_handler);

  unsigned num_inputs = num_args -1;

  // configure the "pipeline"

  input_files.resize(num_inputs);
  inputs.resize (num_inputs);
  for (unsigned i=0; i<num_inputs; i++)
  {
    input_files[i] = string(argv[optind + i]);
    inputs[i] = new spip::ContainerFileRead(input_files[i]);
  }

  spip::ContainerRAM * combined = new spip::ContainerRAM();
  spip::PolCombineRAM * combiner = new spip::PolCombineRAM();

  combiner->set_verbose (verbose);
  for (unsigned i=0; i<num_inputs; i++)
    combiner->add_input (inputs[i]);
  combiner->set_output (combined);
  combiner->set_verbose (verbose);

  output_file = string(argv[optind + num_inputs]);
  spip::ContainerFileWrite * output = new spip::ContainerFileWrite("/tmp");
  output->set_filename (output_file);

  string output_dir = string(".");
  spip::AdaptiveFilterRAM * filter = new spip::AdaptiveFilterRAM(output_dir);
  filter->set_input (combined);
  filter->set_output (output);
  filter->set_verbose (verbose);
  filter->set_filtering (1);

  // open the input file, reading the header
  if (verbose)
    cerr << "main: inputs->open_file()" << endl;
  for (unsigned i=0; i<num_inputs; i++)
    inputs[i]->open_file ();

  // read the header input parameters, determine the container size
  if (verbose)
    cerr << "main: inputs->read_header()" << endl;
  for (unsigned i=0; i<num_inputs; i++)
    inputs[i]->process_header();

  // configure the transform
  if (verbose)
    cerr << "main: combiner->configure(" << inputs[0]->get_order() << ")" << endl;
  combiner->configure(inputs[0]->get_order());

  if (verbose)
    cerr << "main: filter->configure(" << inputs[0]->get_order() << ")" << endl;
  filter->configure(inputs[0]->get_order());

  output->set_order (spip::Ordering::TSPF);

  // configure the output 
  if (verbose)
    cerr << "main: output->process_header()" << endl;
  output->process_header ();
  output->set_order (spip::Ordering::SFPT);

  // read data from the file
  if (verbose)
    cerr << "main: inputs->read_data()" << endl;
  for (unsigned i=0; i<num_inputs; i++)
    inputs[i]->read_data();

  if (verbose)
    cerr << "main: inputs->close_file()" << endl;
  for (unsigned i=0; i<num_inputs; i++)
    inputs[i]->close_file ();

  // transform the data
  if (verbose)
    cerr << "main: combiner->prepare()" << endl;
  combiner->prepare ();

  if (verbose)
    cerr << "main: combiner->combination()" << endl;
  combiner->combination();

  if (verbose)
    cerr << "main: filter->prepare()" << endl;
  filter->prepare();
  if (verbose)
    cerr << "main: filter->combination()" << endl;
  filter->transformation();

  output->set_order (spip::Ordering::TSPF);

  // write the output data
  if (verbose)
    cerr << "main: output->write()" << endl;
  output->write ();

  if (verbose)
    cerr << "main: output->close_file()" << endl;
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
  cout << "uwb_adaptive_filter_combine [options] in_files out_file" << endl;
  cout << " in_files   input PSRDada files" << endl;
  cout << " out_file   output PSRDada file" << endl;
  cout << " -b core    bind to CPU core" << endl;
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
