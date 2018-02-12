/***************************************************************************
 *
 *    Copyright (C) 2015 by Andrew Jameson
 *    Licensed under the Academic Free License version 2.1
 *
 ****************************************************************************/

#include "spip/UDPReceiveMergeDBs.h"
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
#include <sstream>

void usage();
void signal_handler (int signal_value);

spip::UDPReceiveMergeDBs * udpmergedbs;
char quit_threads = 0;

using namespace std;

int main(int argc, char *argv[])
{
  vector<string> keys;

  string format = "simple";

  spip::AsciiHeader config;

  char * config_file = 0;

  // core on which to bind thread operations
  string cores = "-1,-1";

  int control_port = -1;

  int nsecs = -1;

  // control socket for the control port
  spip::TCPSocketServer * ctrl_sock = 0;

  int verbose = 0;

  opterr = 0;
  int c;

  while ((c = getopt(argc, argv, "b:c:f:hk:t:v")) != EOF) 
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

      case 'k':
        keys.append(string(optarg));
        break;

      case 't':
        nsecs = atoi(optarg);
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

  try
  {
    udpmergedbs = new spip::UDPReceiveMergeDBs(keys);

    if (format.compare("simple") == 0)
      udpmergedbs->set_formats (new spip::UDPFormatMeerKATSimple(), new spip::UDPFormatMeerKATSimple());
  #ifdef HAVE_SPEAD2
    else if (format.compare("spead") == 0)
      udpmergedbs->set_formats (new spip::UDPFormatMeerKATSPEAD(), new spip::UDPFormatMeerKATSPEAD());
  #endif
    else
    {
      cerr << "ERROR: unrecognized UDP format [" << format << "]" << endl;
      delete udpmergedbs;
      return (EXIT_FAILURE);
    }

    // Check arguments
    if ((argc - optind) != 1) 
    {
      fprintf(stderr,"ERROR: 1 command line argument expected\n");
      usage();
      return EXIT_FAILURE;
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
    config.load_from_file (argv[optind]);

    if (verbose)
      cerr << "meerkat_udpmergedbs: configuring using fixed config" << endl;
    udpmergedbs->configure (config.raw());

    if (control_port > 0)
    {
      // open a listening socket for observation parameters
      cerr << "meerkat_udpmergedbs: start_control_thread (" << control_port << ")" << endl;
      udpmergedbs->start_control_thread (control_port);

      while (!quit_threads)
      {
        if (verbose)
          cerr << "meerkat_udpmergedbs: starting threads" << endl;
        udpmergedbs->start_threads (core1, core2);
        udpmergedbs->join_threads ();
        if (verbose)
          cerr << "meerkat_udpmergedbs: threads ended" << endl;
        udpmergedbs->set_control_cmd (spip::None);
      }
      udpmergedbs->stop_control_thread ();
    }
    else
    {
      if (verbose)
        cerr << "meerkat_udpmergedbs: receiving" << endl;
      udpmergedbs->start_threads (core1, core2);

      if (verbose)
        cerr << "meerkat_udpmergedbs: opening data block" << endl;
      if (udpmergedbs->open ())
      {
        cerr << "meerkat_udpmergedbs: issuing start command" << endl;
        udpmergedbs->set_control_cmd (spip::Start);
      }
      else
      {
        if (verbose)
          cerr << "meerkat_udpmergedbs: failed to open data stream" << endl;
      }
      udpmergedbs->join_threads ();
    }

    quit_threads = 1;
    delete udpmergedbs;
  }
  catch (std::exception& exc)
  {
    cerr << "meerkat_udpmergedbs: ERROR: " << exc.what() << endl;
    return -1;
  }


  return 0;
}

void usage() 
{
  cout << "meerkat_udpmergedbs [options] header\n"
      "  header      ascii file contain header\n"
      "  -b c1,c2    bind pols 1 and 2 to cores c1 and c2\n"
      "  -c port     listen for control commands on port\n"
  #ifdef HAVE_SPEAD2
      "  -f format   UDP data format [simple spead]\n"
  #else
      "  -f format   UDP data format [simple]\n"
  #endif
      "  -t sec      Receive data for sec seconds\n"
      "  -k key      shared memory keys to write to [default " << std::hex << DADA_DEFAULT_BLOCK_KEY << std::dec << "]\n"
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
  udpmergedbs->set_control_cmd (spip::Quit);
}
