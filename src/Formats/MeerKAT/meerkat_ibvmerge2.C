/***************************************************************************
 *
 *    Copyright (C) 2015 by Andrew Jameson
 *    Licensed under the Academic Free License version 2.1
 *
 ****************************************************************************/

#include "spip/IBVReceiverMerge2.h"
#include "spip/UDPFormatMeerKATSPEAD.h"
#include "spip/UDPFormatMeerKATSPEAD1k.h"
#include "spip/TCPSocketServer.h"

#include <unistd.h>
#include <signal.h>
#include <pthread.h>

#include <cstdio>
#include <cstring>
#include <iostream>
#include <sstream>

void usage();
void signal_handler (int signal_value);

spip::IBVReceiverMerge2 * ibvmerge2;
char quit_threads = 0;

using namespace std;

int main(int argc, char *argv[])
{
  string format = "spead";

  spip::AsciiHeader config;

  // core on which to bind thread operations
  string cores = "-1,-1";

  int control_port = -1;

  int verbose = 0;

  opterr = 0;
  int c;

  while ((c = getopt(argc, argv, "b:c:f:hv")) != EOF) 
  {
    switch(c) 
    {
      case 'b':
        cores = string(optarg);
        break;

      case 'c':
        control_port = atoi(optarg);
        break;

      case 'f':
        format = optarg;
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
  if ((argc - optind) != 1)
  {
    fprintf(stderr,"ERROR: 1 command line argument expected\n");
    usage();
    return EXIT_FAILURE;
  }

  string config_file = string(argv[optind]);
 
  try
  {
    boost::asio::io_service io_service;

    ibvmerge2 = new spip::IBVReceiverMerge2(io_service);

    if (format.compare("spead") == 0)
      ibvmerge2->set_formats (new spip::UDPFormatMeerKATSPEAD(), new spip::UDPFormatMeerKATSPEAD());
    else if (format.compare("spead1k") == 0)
      ibvmerge2->set_formats (new spip::UDPFormatMeerKATSPEAD1k(), new spip::UDPFormatMeerKATSPEAD1k());
    else
    {
      cerr << "ERROR: unrecognized UDP format [" << format << "]" << endl;
      delete ibvmerge2;
      return (EXIT_FAILURE);
    }

    int core1, core2;
    string delimited = ",";
    size_t pos = cores.find(delimited);
    string str1 = cores.substr(0, pos);
    string str2 = cores.substr(pos+1, cores.length());
    istringstream(str1) >> core1;
    istringstream(str2) >> core2;

    signal(SIGINT, signal_handler);
   
    // config for the this data stream
    config.load_from_file (config_file.c_str());

    if (verbose)
      cerr << "meerkat_ibvmerge2: configuring using fixed config" << endl;
    ibvmerge2->configure (config.raw());

    if (control_port > 0)
    {
      // open a listening socket for observation parameters
      cerr << "meerkat_ibvmerge2: start_control_thread (" << control_port << ")" << endl;
      ibvmerge2->start_control_thread (control_port);

      while (!quit_threads)
      {
        if (verbose)
          cerr << "meerkat_ibvmerge2: starting threads" << endl;
        ibvmerge2->start_threads (core1, core2);
        ibvmerge2->join_threads ();
        if (verbose)
          cerr << "meerkat_ibvmerge2: threads ended" << endl;
        ibvmerge2->set_control_cmd (spip::None);
      }
      ibvmerge2->stop_control_thread ();
    }
    else
    {
      if (verbose)
        cerr << "meerkat_ibvmerge2: receiving" << endl;
      ibvmerge2->start_threads (core1, core2);

      if (verbose)
        cerr << "meerkat_ibvmerge2: opening data block" << endl;
      if (ibvmerge2->open ())
      {
        cerr << "meerkat_ibvmerge2: issuing start command" << endl;
        ibvmerge2->set_control_cmd (spip::Start);
      }
      else
      {
        if (verbose)
          cerr << "meerkat_ibvmerge2: failed to open data stream" << endl;
      }
      ibvmerge2->join_threads ();
    }

    quit_threads = 1;
    delete ibvmerge2;
  }
  catch (std::exception& exc)
  {
    cerr << "meerkat_ibvmerge2: ERROR: " << exc.what() << endl;
    return -1;
  }


  return 0;
}

void usage() 
{
  cout << "meerkat_ibvmerge2 [options] header\n"
      "  header      ascii file contain header\n"
      "  -b c1,c2    bind pols 1 and 2 to cores c1 and c2\n"
      "  -c port     listen for control commands on port\n"
      "  -f format   UDP data format [spead spead1k]\n"
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
  ibvmerge2->set_control_cmd (spip::Quit);
}
