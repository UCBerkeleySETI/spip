/***************************************************************************
 *
 *    Copyright (C) 2015 by Andrew Jameson
 *    Licensed under the Academic Free License version 2.1
 *
 ****************************************************************************/

#include "spip/UDPReceiveMerge2DB.h"
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

spip::UDPReceiveMerge2DB * udpmerge2db;
char quit_threads = 0;

using namespace std;

int main(int argc, char *argv[])
{
  string format = "simple";

  spip::AsciiHeader config;

  // core on which to bind thread operations
  string cores = "-1,-1";

  int control_port = -1;

  int nsecs = -1;

  // control socket for the control port
  spip::TCPSocketServer * ctrl_sock = 0;

  int verbose = 0;

  opterr = 0;
  int c;

  while ((c = getopt(argc, argv, "b:c:f:ht:v")) != EOF) 
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

  // Check arguments
  if ((argc - optind) != 3)
  {
    fprintf(stderr,"ERROR: 3 command line argument expected\n");
    usage();
    return EXIT_FAILURE;
  }

  string config_file = string(argv[optind]);
  string key1 = string(argv[optind+1]);
  string key2 = string(argv[optind+2]);
 
  try
  {
    udpmerge2db = new spip::UDPReceiveMerge2DB(key1.c_str(), key2.c_str());

    if (format.compare("simple") == 0)
      udpmerge2db->set_formats (new spip::UDPFormatMeerKATSimple(), new spip::UDPFormatMeerKATSimple());
  #ifdef HAVE_SPEAD2
    else if (format.compare("spead") == 0)
      udpmerge2db->set_formats (new spip::UDPFormatMeerKATSPEAD(), new spip::UDPFormatMeerKATSPEAD());
  #endif
    else
    {
      cerr << "ERROR: unrecognized UDP format [" << format << "]" << endl;
      delete udpmerge2db;
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
      cerr << "meerkat_udpmerge2db: configuring using fixed config" << endl;
    udpmerge2db->configure (config.raw());

    if (control_port > 0)
    {
      // open a listening socket for observation parameters
      cerr << "meerkat_udpmerge2db: start_control_thread (" << control_port << ")" << endl;
      udpmerge2db->start_control_thread (control_port);

      while (!quit_threads)
      {
        if (verbose)
          cerr << "meerkat_udpmerge2db: starting threads" << endl;
        udpmerge2db->start_threads (core1, core2);
        udpmerge2db->join_threads ();
        if (verbose)
          cerr << "meerkat_udpmerge2db: threads ended" << endl;
        udpmerge2db->set_control_cmd (spip::None);
      }
      udpmerge2db->stop_control_thread ();
    }
    else
    {
      if (verbose)
        cerr << "meerkat_udpmerge2db: receiving" << endl;
      udpmerge2db->start_threads (core1, core2);

      if (verbose)
        cerr << "meerkat_udpmerge2db: opening data block" << endl;
      if (udpmerge2db->open ())
      {
        cerr << "meerkat_udpmerge2db: issuing start command" << endl;
        udpmerge2db->set_control_cmd (spip::Start);
      }
      else
      {
        if (verbose)
          cerr << "meerkat_udpmerge2db: failed to open data stream" << endl;
      }
      udpmerge2db->join_threads ();
    }

    quit_threads = 1;
    delete udpmerge2db;
  }
  catch (std::exception& exc)
  {
    cerr << "meerkat_udpmerge2db: ERROR: " << exc.what() << endl;
    return -1;
  }


  return 0;
}

void usage() 
{
  cout << "meerkat_udpmerge2db [options] header key1 key2\n"
      "  header      ascii file contain header\n"
      "  key1        shared memory key for subband 1\n"
      "  key2        shared memory key for subband 2\n"
      "  -b c1,c2    bind pols 1 and 2 to cores c1 and c2\n"
      "  -c port     listen for control commands on port\n"
  #ifdef HAVE_SPEAD2
      "  -f format   UDP data format [simple spead]\n"
  #else
      "  -f format   UDP data format [simple]\n"
  #endif
      "  -t sec      Receive data for sec seconds\n"
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
  udpmerge2db->set_control_cmd (spip::Quit);
}
