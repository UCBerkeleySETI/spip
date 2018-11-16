/***************************************************************************
 *
 *    Copyright (C) 2015 by Andrew Jameson
 *    Licensed under the Academic Free License version 2.1
 *
 ****************************************************************************/

#include "spip/IBVReceiveDB.h"
#include "spip/UDPFormatMeerKATSPEAD.h"
#include "spip/UDPFormatMeerKATSPEAD1k.h"
#include "spip/TCPSocketServer.h"

#include <unistd.h>
#include <signal.h>
#include <pthread.h>

#include <cstdio>
#include <cstring>
#include <iostream>
#include <stdexcept>

void usage();
void signal_handler (int signal_value);

spip::IBVReceiveDB * ibvdb;
char quit_threads = 0;

using namespace std;

int main(int argc, char *argv[]) try
{
  string key = "dada";

  string * format = new string("spead");

  spip::AsciiHeader config;

  // tcp control port to receive configuration
  int control_port = -1;

  // control socket for the control port
  spip::TCPSocketServer * ctrl_sock = 0;

  int verbose = 0;

  opterr = 0;
  int c;

  int core = -1;

  while ((c = getopt(argc, argv, "b:c:f:hk:v")) != EOF) 
  {
    switch(c) 
    {
      case 'b':
        core = atoi(optarg);
        break;

      case 'c':
        control_port = atoi(optarg);
        break;

      case 'f':
        format = new string(optarg);
        break;

      case 'k':
        key = optarg;
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

  boost::asio::io_service io_service;

  // create a UDP recevier that writes to a data block
  ibvdb = new spip::IBVReceiveDB (key.c_str(), io_service);
  ibvdb->set_verbosity (verbose);

  if (format->compare("spead") == 0)
    ibvdb->set_format (new spip::UDPFormatMeerKATSPEAD());
  else if (format->compare("spead1k") == 0)
    ibvdb->set_format (new spip::UDPFormatMeerKATSPEAD1k());
  else
  {
    cerr << "ERROR: unrecognized UDP format [" << format << "]" << endl;
    delete ibvdb;
    return (EXIT_FAILURE);
  }

  // Check arguments
  if ((argc - optind) != 1) 
  {
    fprintf(stderr,"ERROR: 1 command line argument expected\n");
    usage();
    return EXIT_FAILURE;
  }
 
  signal(SIGINT, signal_handler);

  // config for the this data stream
  config.load_from_file (argv[optind]);

  uint64_t data_bufsz = ibvdb->get_data_bufsz();
  if (config.set("RESOLUTION", "%lu", data_bufsz) < 0)
  {
    fprintf (stderr, "ERROR: could not write RESOLUTION=%lu to config\n", data_bufsz);
    return (EXIT_FAILURE);
  }

  if (verbose)
    cerr << "meerkat_ibvdb: configuring using fixed config" << endl;
  ibvdb->configure (config.raw());

  if (verbose)
    cerr << "meerkat_ibvdb: starting stats thread" << endl;
  ibvdb->start_stats_thread ();

  if (control_port > 0)
  {
    // open a listening socket for observation parameters
    cerr << "meerkat_ibvdb: start_control_thread (" << control_port << ")" << endl;
    ibvdb->start_control_thread (control_port);

    while (!quit_threads)
    {
      if (verbose)
        cerr << "meerkat_ibvdb: ibvdb->receive" << endl;

      bool result = ibvdb->receive (core);
      if (!result)
      {
        cerr << "meerkat_ibvdb: receive failed, exiting" << endl;
        quit_threads = 1;
      }  

      if (verbose)
        cerr << "meerkat_ibvdb: receive returned" << endl;
      ibvdb->set_control_cmd (spip::None);
    }

    // ensure the control thread is stopped
    ibvdb->stop_control_thread();

  }
  else
  {
    if (verbose)
      cerr << "meerkat_ibvdb: writing header to data block" << endl;
    ibvdb->open ();

    if (verbose)
      cerr << "meerkat_ibvdb: issuing start command" << endl;
    ibvdb->start_capture ();

    if (verbose)
      cerr << "meerkat_ibvdb: calling receive" << endl;
    ibvdb->receive (core);
  
    ibvdb->close();
  }

  ibvdb->stop_stats_thread ();

  delete ibvdb;
}
catch (std::exception& exc)
{
  cerr << "ERROR: " << exc.what() << endl;
  return -1;
  return 0;
}

void usage() 
{
  cout << "meerkat_ibvdb [options] config\n"
    "  config      ascii file containing fixed configuration\n"
    "  -b core     bind computation to specified CPU core\n"
    "  -c port     control port for dynamic configuration\n"
    "  -f format   UDP data format [spead spead1k]\n"
    "  -h          print this help text\n"
    "  -k key      PSRDada shared memory key to write to [default " << std::hex << DADA_DEFAULT_BLOCK_KEY << "]\n"
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
  ibvdb->stop_capture();
}

