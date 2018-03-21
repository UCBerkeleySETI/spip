/***************************************************************************
 *
 *    Copyright (C) 2015 by Andrew Jameson
 *    Licensed under the Academic Free License version 2.1
 *
 ****************************************************************************/

#include "spip/TCPDummyServer.h"

#include <unistd.h>
#include <signal.h>
#include <pthread.h>

#include <cstdio>
#include <cstring>
#include <iostream>
#include <stdexcept>

void usage();
void signal_handler (int signal_value);

spip::TCPDummyServer * tcpdummy;
char quit_threads = 0;

using namespace std;

int main(int argc, char *argv[]) try
{
  // tcp control port to serve configuration
  int port = -1;

  int verbose = 0;

  opterr = 0;
  int c;

  int core = -1;

  while ((c = getopt(argc, argv, "b:c:hv")) != EOF) 
  {
    switch(c) 
    {
      case 'b':
        core = atoi(optarg);
        break;

      case 'c':
        port = atoi(optarg);
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

  if (port == -1)
  {
    cerr << "ERROR: a control port must be specified" << endl;
    usage();
    return EXIT_FAILURE;
  }

  // create a UDP recevier that writes to a data block
  tcpdummy = new spip::TCPDummyServer ();

  if (verbose)
    cerr << "meerkat_tcpdummy: tcpdummy->set_verbosity(" << verbose << ")" << endl;
  tcpdummy->set_verbosity (verbose);

  // Check arguments
  if ((argc - optind) != 0) 
  {
    fprintf(stderr,"ERROR: 0 command line arguments expected\n");
    usage();
    return EXIT_FAILURE;
  }
 
  signal(SIGINT, signal_handler);

  while (!quit_threads)
  {
    if (verbose)
      cerr << "meerkat_tcpdummy: tcpdummy->receive(" << port << ")" << endl;

    bool result = tcpdummy->serve (port);
    if (!result)
    {
      cerr << "meerkat_tcpdummy: serve failed, exiting" << endl;
      quit_threads = 1;
    }  

    if (verbose)
      cerr << "meerkat_tcpdummy: serve returned" << endl;
  } 

  delete tcpdummy;
}
catch (std::exception& exc)
{
  cerr << "ERROR: " << exc.what() << endl;
  return -1;
  return 0;
}

void usage() 
{
  cout << "meerkat_tcpdummy [options]\n"
    "  -b core     bind computation to specified CPU core\n"
    "  -c port     control port for dynamic configuration\n"
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
  tcpdummy->set_control_cmd (spip::Quit);
}
