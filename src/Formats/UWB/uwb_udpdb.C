/***************************************************************************
 *
 *    Copyright (C) 2015 by Andrew Jameson
 *    Licensed under the Academic Free License version 2.1
 *
 ****************************************************************************/

#include "spip/UDPReceiveDB.h"
#include "spip/UDPFormatVDIF.h"
#include "spip/UDPFormatDualVDIF.h"
#include "spip/UDPFormatUWB.h"

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

  string * format_name = new string("vdif");

  spip::AsciiHeader config;

  // tcp control port to receive configuration
  int control_port = -1;

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
        format_name = new string(optarg);
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

  // configure the UDP format as VDIF or UWB
  spip::UDPFormatVDIF * format;
  if (format_name->compare("uwb") == 0)
    format = new spip::UDPFormatUWB();
  else if (format_name->compare("vdif") == 0)
    format = new spip::UDPFormatVDIF();
  else if (format_name->compare("dualvdif") == 0)
    format = new spip::UDPFormatDualVDIF();
  else
  {
    cerr << "ERROR: unrecognized UDP format [" << format << "]" << endl;
    delete udpdb;
    return (EXIT_FAILURE);
  }
  format->set_self_start (control_port == -1);
  udpdb->set_format (format);

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

  unsigned resolution = 16384;
  if (config.set("RESOLUTION", "%u", resolution) < 0)
  {
    fprintf (stderr, "ERROR: could not write RESOLUTION=%u to config\n", resolution);
    return (EXIT_FAILURE);
  }

  if (verbose)
    cerr << "uwb_udpdb: configuring using fixed config" << endl;
  udpdb->configure (config.raw());

  // prepare a header which combines config with observation parameters
  spip::AsciiHeader header;
  header.load_from_str (config.raw());

  udpdb->start_stats_thread ();

  if (control_port > 0)
  {
    // open a listening socket for observation parameters
    cerr << "uwb_udpdb: start_control_thread (" << control_port << ")" << endl;
    udpdb->start_control_thread (control_port);

    bool keep_receiving = true;
    while (keep_receiving)
    {
      // start the main receiving thread to receive 1 observation of data
      if (verbose)
        cerr << "uwb_udpdb: receiving" << endl;
      keep_receiving = udpdb->receive (core);

      // reset the control command
      if (verbose)
        cerr << "uwb_udpdb: udpdb->set_control_cmd (None)" << endl;
      udpdb->set_control_cmd (spip::None);

    }
  }
  else
  {
    if (verbose)
      cerr << "uwb_udpdb: writing header to data block" << endl;
    udpdb->open ();

    cerr << "uwb_udpdb: issuing start command" << endl;
    udpdb->start_capture ();

    cerr << "uwb_udpdb: calling receive" << endl;
    udpdb->receive (core);
  }

  udpdb->stop_stats_thread ();
  udpdb->close();

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
  cout << "uwb_udpdb [options] config\n"
    "  config      ascii file containing fixed configuration\n"
    "  -b core     bind computation to specified CPU core\n"
    "  -c port     control port for dynamic configuration\n"
    "  -f format   UDP format to use: vdif or dualvdif [default vdif]\n"
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

