/***************************************************************************
 *
 *    Copyright (C) 2015 by Andrew Jameson
 *    Licensed under the Academic Free License version 2.1
 *
 ****************************************************************************/

#include "spip/UDPReceiveDB.h"
#include "spip/UDPFormatMeerKATSimple.h"
#ifdef HAVE_SPEAD2
#include "spip/UDPFormatMeerKATSPEAD.h"
#endif
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

spip::UDPReceiveDB * udpdb;
char quit_threads = 0;

using namespace std;

int main(int argc, char *argv[]) try
{
  string key = "dada";

  string * format = new string("simple");

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

  // create a UDP recevier that writes to a data block
  udpdb = new spip::UDPReceiveDB (key.c_str());

  if (format->compare("simple") == 0)
    udpdb->set_format (new spip::UDPFormatMeerKATSimple());
#ifdef HAVE_SPEAD2
  else if (format->compare("spead") == 0)
    udpdb->set_format (new spip::UDPFormatMeerKATSPEAD());
#endif
  else
  {
    cerr << "ERROR: unrecognized UDP format [" << format << "]" << endl;
    delete udpdb;
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

  uint64_t data_bufsz = udpdb->get_data_bufsz();
  if (config.set("RESOLUTION", "%lu", data_bufsz) < 0)
  {
    fprintf (stderr, "ERROR: could not write RESOLUTION=%lu to config\n", data_bufsz);
    return (EXIT_FAILURE);
  }

  if (verbose)
    cerr << "meerkat_udpdb: configuring using fixed config" << endl;
  udpdb->configure (config.raw());

  if (verbose)
    cerr << "meerkat_udpdb: starting stats thread" << endl;
  //udpdb->start_stats_thread ();

  if (control_port > 0)
  {
    // open a listening socket for observation parameters
    cerr << "meerkat_udpdb: start_control_thread (" << control_port << ")" << endl;
    udpdb->start_control_thread (control_port);

    while (!quit_threads)
    {
      if (verbose)
        cerr << "meerkat_udpdb: udpdb->receive" << endl;

      bool result = udpdb->receive (core);
      if (!result)
      {
        cerr << "meerkat_udpdb: receive failed, exiting" << endl;
        quit_threads = 1;
      }  

      if (verbose)
        cerr << "meerkat_udpdb: receive returned" << endl;
      udpdb->set_control_cmd (spip::None);
    }

    // ensure the control thread is stopped
    udpdb->stop_control_thread();

  }
  else
  {
    if (verbose)
      cerr << "meerkat_udpdb: writing header to data block" << endl;
    udpdb->open ();

    if (verbose)
      cerr << "meerkat_udpdb: issuing start command" << endl;
    udpdb->start_capture ();

    if (verbose)
      cerr << "meerkat_udpdb: calling receive" << endl;
    udpdb->receive (core);
  
    udpdb->close();

  }

  //
  udpdb->stop_stats_thread ();

  delete udpdb;
}
catch (std::exception& exc)
{
  cerr << "ERROR: " << exc.what() << endl;
  return -1;
  return 0;
}

void usage() 
{
  cout << "meerkat_udpdb [options] config\n"
    "  config      ascii file containing fixed configuration\n"
    "  -b core     bind computation to specified CPU core\n"
    "  -c port     control port for dynamic configuration\n"
#ifdef HAVE_SPEAD2
    "  -f format   UDP data format [simple spead]\n"
#else
    "  -f format   UDP data format [simple]\n"
#endif
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

  udpdb->stop_capture();
}

