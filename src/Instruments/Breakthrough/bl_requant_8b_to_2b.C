/***************************************************************************
 *
 *   Copyright (C) 2018 Andrew Jameson
 *   Licensed under the Academic Free License version 2.1
 *
 ***************************************************************************/

#include "spip/HardwareAffinity.h"
#include "spip/BlRequant8i2uClient.h"

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
  string read_key;
  string write_key;

  spip::HardwareAffinity hw_affinity;

  int verbose = 0;

  int core = -1;

  opterr = 0;
  int c;

  while ((c = getopt(argc, argv, "b:hk:v")) != EOF)
  {
    switch(c)
    {
      case 'b':
        core = atoi(optarg);
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

  // check command line arguments
  int nargs = argc - optind;
  if (nargs != 2)
  {
    cerr << "Error: 2 command line arguments are required" << endl;
    usage();
    return (EXIT_FAILURE);
  }

  if (core >= 0)
  {
    hw_affinity.bind_process_to_cpu_core (core);
  }

  read_key = std::string(argv[optind+0]);
  write_key = std::string(argv[optind+1]);
  spip::BlRequant8i2uClient * op;

  op = new spip::BlRequant8i2uClient (read_key.c_str(),
                                              write_key.c_str());
  if (verbose)
    cerr << "bl_requant_8b_to_2b: main()" << endl;
  try
  {
    op->main();
  }
  catch (std::exception& exc)
  {
    cerr << "bl_requant_8b_to_2b: ERROR: " << exc.what() << endl;
    return -1;
  }

  delete op;
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
  cout << "bl_requant_8b_to_2b [options] input out_key\n"
    "  in_key      PSRDada shared memory key for input\n"
    "  out_key     PSRDada shared memory key for output\n"
    "  -b core     bind computation to specified CPU core\n"
    "  -h          print this help text\n"
    "  -v          verbose output\n"
    "\n"
    "  converts 16-bit offset binary integer input to two's complement 8-bit output\n"
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
