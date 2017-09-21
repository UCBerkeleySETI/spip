/***************************************************************************
 *
 *   Copyright (C) 2016 Andrew Jameson
 *   Licensed under the Academic Free License version 2.1
 *
 ***************************************************************************/

#include "spip/AsciiHeader.h"

#include <sys/types.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <unistd.h>
#include <errno.h>

#include <cstdarg>
#include <cstring>
#include <cstdlib>
#include <cstdio>
#include <iostream>
#include <stdexcept>

using namespace std;

int main () try
{
  spip::AsciiHeader config;
  spip::AsciiHeader header;

  config.load_from_file ("config");

  cerr << "Raw header:" << endl;
  cerr << config.raw() << endl;

  header.clone (config);

  const char * cmds = "COMMAND             START\n\
SOURCE              J1644-4559_R\n\
RA                  None\n\
DEC                 None\n\
TOBS                None\n\
OBSERVER            Sarah\n\
PID                 None\n\
MODE                CAL\n\
CALFREQ             11.1111111111\n\
OBS_OFFSET          0\n\
PERFORM_FOLD        1\n\
PERFORM_SEARCH      0\n\
PERFORM_TRANS       0\n\
ADC_SYNC_TIME       1501586468.0\n\
ANTENNAE            m006\n\
SCHEDULE_BLOCK_ID   20170801-0030\n\
EXPERIMENT_ID       20170801-0030\n\
PROPOSAL_ID         None\n\
PROGRAM_BLOCK_ID    TBD\n\
DESCRIPTION         Noise diode test\n\
UTC_START           2017-08-01-15:53:29\n";

  header.append_from_str (cmds);

  header.del ("COMMAND");

  uint64_t picoseconds = 1034168256;
  header.set("PICOSECONDS","%lu", picoseconds);
  header.set("PICOSECONDS","%lu", picoseconds);

  cerr << "header:" << endl;
  cerr << "[" << header.raw() << "]" << endl;

  return 0;
}
catch (std::exception &exc)
{
  cerr << "test_AsciiHeader: ERROR " << exc.what() << endl;
  return -1;
}
