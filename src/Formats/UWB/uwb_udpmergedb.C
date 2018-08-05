/***************************************************************************
 *
 *    Copyright (C) 2015 by Andrew Jameson
 *    Licensed under the Academic Free License version 2.1
 *
 ****************************************************************************/

#include "spip/UDPReceiveMergeDB.h"
#include "spip/UDPFormatVDIF.h"
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

spip::UDPReceiveMergeDB * udpmergedb;
char quit_threads = 0;

using namespace std;

int main(int argc, char *argv[])
{
  string key = "dada";

  spip::AsciiHeader config;

  // core on which to bind thread operations
  string cores = "-1,-1";

  int control_port = -1;

  int verbose = 0;

  opterr = 0;
  int c;

  while ((c = getopt(argc, argv, "b:c:hk:v")) != EOF) 
  {
    switch(c) 
    {
      case 'b':
        cores = string(optarg);
        break;

      case 'c':
        control_port = atoi(optarg);
        break;

      case 'h':
        cerr << "Usage: " << endl;
        usage();
        exit(EXIT_SUCCESS);
        break;

      case 'k':
        key = optarg;
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

    udpmergedb = new spip::UDPReceiveMergeDB(key.c_str());
    udpmergedb->set_formats (new spip::UDPFormatVDIF(), new spip::UDPFormatVDIF());

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
      cerr << "uwb_udpmergedb: configuring using fixed config" << endl;
    udpmergedb->configure (config.raw());

    if (control_port > 0)
    {
      // open a listening socket for observation parameters
      cerr << "uwb_udpmergedb: start_control_thread (" << control_port << ")" << endl;

      udpmergedb->start_control_thread (control_port);

      while (!quit_threads)
      {
        if (verbose)
          cerr << "uwb_udpmergedb: starting threads" << endl;
        udpmergedb->start_threads (core1, core2);
        udpmergedb->join_threads ();
        if (verbose)
          cerr << "uwb_udpmergedb: threads ended" << endl;
        udpmergedb->set_control_cmd (spip::None);
      }
      udpmergedb->stop_control_thread ();
    }
    else
    {
      if (verbose)
        cerr << "uwb_udpmergedb: receiving" << endl;
      udpmergedb->start_threads (core1, core2);

      if (verbose)
        cerr << "uwb_udpmergedb: opening data block" << endl;
      if (udpmergedb->open ())
      {
        cerr << "uwb_udpmergedb: issuing start command" << endl;
        udpmergedb->set_control_cmd (spip::Start);
      }
      else
      {
        if (verbose)
          cerr << "uwb_udpmergedb: failed to open data stream" << endl;
      }
      udpmergedb->join_threads ();
    }

    quit_threads = 1;
    delete udpmergedb;
  }
  catch (std::exception& exc)
  {
    cerr << "uwb_udpmergedb: ERROR: " << exc.what() << endl;
    return -1;
  }


  return 0;
}

void usage() 
{
  cout << "uwb_udpmergedb [options] header\n"
      "  header      ascii file contain header\n"
      "  -b c1,c2    bind pols 1 and 2 to cores c1 and c2\n"
      "  -c port     listen for control commands on port\n"
      "  -t sec      Receive data for sec seconds\n"
      "  -k key      shared memory key to write to [default " << std::hex << DADA_DEFAULT_BLOCK_KEY << std::dec << "]\n"
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
  udpmergedb->set_control_cmd (spip::Quit);
}
