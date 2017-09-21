/***************************************************************************
 *
 *   Copyright (C) 2016 Andrew Jameson
 *   Licensed under the Academic Free License version 2.1
 *
 ***************************************************************************/

#include "spip/Container.h"

#include <iostream>
#include <stdexcept>

using namespace std;

//! Global verbosity flag
bool spip::Container::verbose = false;

spip::Container::Container ()
{
  ndat = 1;
  nchan = 1;
  nsignal = 1;
  ndim = 1;
  npol = 1;
  nbit = 8;
  size = 0;

  buffer = NULL;
}

spip::Container::~Container ()
{
}

void spip::Container::read_header()
{
  if (header.get ("NANT", "%u", &nsignal) != 1)
    throw invalid_argument ("NSIGNAL did not exist in header");

  if (header.get ("NCHAN", "%u", &nchan) != 1)
    throw invalid_argument ("NCHAN did not exist in header");

  if (header.get ("NBIT", "%u", &nbit) != 1)
    throw invalid_argument ("NBIT did not exist in header");

  if (header.get ("NPOL", "%u", &npol) != 1)
    throw invalid_argument ("NPOL did not exist in header");

  if (header.get ("NDIM", "%u", &ndim) != 1)
    throw invalid_argument ("NDIM did not exist in header");

  if (header.get ("TSAMP", "%lf", &tsamp) != 1)
    throw invalid_argument ("TSAMP did not exist in header");

  if (header.get ("BW", "%lf", &bandwidth) != 1)
    throw invalid_argument ("BW did not exist in header");

  if (header.get ("FREQ", "%lf", &centre_freq) != 1)
    throw invalid_argument ("FREQ did not exist in header");

  char tmp_buf[128];
  if (header.get ("UTC_START", "%s", tmp_buf) == -1)
    throw invalid_argument ("failed to read UTC_START from header");

  // parse UTC_START into spip::Time
  utc_start = new spip::Time(tmp_buf);

  if (header.get ("PICOSECONDS", "%lu", &utc_start_pico) != 1)
    throw invalid_argument ("PICOSECONDS did not exist in header");

  if (header.get ("OBS_OFFSET", "%lu", &obs_offset) != 1)
    throw invalid_argument ("OBS_OFFSET did not exist in header");

  if (header.get ("OSAMP_NUMERATOR", "%u", &(oversampling_ratio[0])) != 1)
    oversampling_ratio[0] = 1;

  if (header.get ("OSAMP_DENOMINATOR", "%u", &(oversampling_ratio[1])) != 1)
    oversampling_ratio[1] = 1;
}

void spip::Container::write_header ()
{
  if (header.set ("NANT", "%u", nsignal) < 0)
    throw invalid_argument ("Could not write NANT to header");

  if (header.set ("NCHAN", "%u", nchan) < 0)
    throw invalid_argument ("Could not write NCHAN to header");

  if (header.set ("NBIT", "%u", nbit) < 0)
    throw invalid_argument ("Could not write NBIT to header");

  if (header.set ("NPOL", "%u", npol) < 0)
    throw invalid_argument ("Could not write NPOL to header");

  if (header.set ("NDIM", "%u", ndim) < 0)
    throw invalid_argument ("Could not write NDIM to header");

  if (header.set ("TSAMP", "%lf", tsamp) < 0)
    throw invalid_argument ("Could not write TSAMP to header");

  if (header.set ("BW", "%lf", bandwidth) < 0)
    throw invalid_argument ("Could not write BW to header");

  if (header.set ("FREQ", "%lf", centre_freq) < 0)
    throw invalid_argument ("Could not write FREQ to header");

  std::string utc_str = utc_start->get_gmtime();
  if (header.set ("UTC_START", "%s", utc_str.c_str()) < 0)
    throw invalid_argument ("Could not write UTC_START to header");

  if (header.set ("PICOSECONDS", "%lu", utc_start_pico) < 0)
    throw invalid_argument ("Could not write PICOSECONDS to header");

  if (header.set ("OBS_OFFSET", "%lu", obs_offset) < 0)
    throw invalid_argument ("Could not write OBS_OFFSET to header");

  if (header.set ("OSAMP_NUMERATOR", "%u", (oversampling_ratio[0])) < 0)
    throw invalid_argument ("Could not write OSAMP_NUMERATOR to header");

  if (header.set ("OSAMP_DENOMINATOR", "%u", (oversampling_ratio[1])) < 0)
    throw invalid_argument ("Could not write OSAMP_DENOMINATOR to header");

}

void spip::Container::clone_header (const spip::AsciiHeader &obj)
{
  header.clone (obj);
}

