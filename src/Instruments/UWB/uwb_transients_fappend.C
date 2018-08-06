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
#include "spip/AppendFrequencyRAM.h"
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
  // input files to append
  vector<string> in_keys;

  string out_key;

  spip::HardwareAffinity hw_affinity;

  int verbose = 0;

  int c;

  opterr = 0;

  int core;

  while ((c = getopt(argc, argv, "b:hk:v")) != EOF)
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

      case 'k':
        in_keys.push_back (optarg);
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
  if (num_args < 1)
  {
    fprintf(stderr,"ERROR: 1 command line argument expected\n");
    usage();
    return EXIT_FAILURE;
  }

  signal(SIGINT, signal_handler);

  out_key = string (arg[optind]);



  spip::ContainerFileWrite * output = new spip::ContainerFileWrite(output_dir);

  spip::AppendFrequencyRAM * appender = new spip::AppendFrequencyRAM();

  appender->set_verbose (verbose);

  spip::ContainerFileRead * input1 = inputs[0];
  spip::ContainerFileRead * input2 = inputs[1];

  for (unsigned i=0; i<num_inputs-1; i++)
  {
    appender->set_input1 (input1);
    appender->set_input1 (input2);
    appender->set_output (output);

  // open the input file, reading the header
  if (verbose)
    cerr << "main: inputs->open_file()" << endl;
  input1->open_file ();
  input2->open_file ();

  // read the header input parameters, determine the container size
  if (verbose)
    cerr << "main: input->read_header()" << endl;
  input1->process_header();
  input2->process_header();

  // configure the transform
  if (verbose)
    cerr << "main: appender->configure()" << endl;
  appender->configure(input1->get_order());

  // configure the output 
  if (verbose)
    cerr << "main: output->process_header()" << endl;
  output->process_header ();

  // read data from the file
  if (verbose)
    cerr << "main: input->read_data()" << endl;
  input1->read_data();
  input2->read_data();

  // transform the data
  if (verbose)
    cerr << "main: appender->prepare()" << endl;
  appender->prepare ();
  if (verbose)
    cerr << "main: appender->transformation ()" << endl;
  appender->cominbation();

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
  cout << "uwb_transients_fappend [options] out_key" << endl;
  cout << " out_key    output ring buffer" << endl;
  cout << " -b core    bind to CPU core" << endl;
  cout << " -h         display usage" << endl;
  cout << " -k in_key  input ring buffers (specify multiple times if required)" << endl;
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
