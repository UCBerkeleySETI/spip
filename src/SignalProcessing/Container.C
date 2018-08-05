/***************************************************************************
 *
 *   Copyright (C) 2016 Andrew Jameson
 *   Licensed under the Academic Free License version 2.1
 *
 ***************************************************************************/

#include "spip/Container.h"
#include "spip/Error.h"

#include <iostream>
#include <stdexcept>
#include <limits> 

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
  nbin = 1;
  nbit = 8;
  size = 0;

  file_size = -1;
  compute_file_size = false;

  buffer = NULL;
}

spip::Container::~Container ()
{
}

void spip::Container::read_header()
{
  if (header.get ("NANT", "%u", &nsignal) != 1)
  {
    if (spip::Container::verbose)
      cerr << "spip::Container::read_header NANT did not exist in header, assuming 1" << endl;
    nsignal = 1;
  }

  if (header.get ("NCHAN", "%u", &nchan) != 1)
    throw invalid_argument ("NCHAN did not exist in header");

  if (header.get ("NBIT", "%u", &nbit) != 1)
    throw invalid_argument ("NBIT did not exist in header");

  if (header.get ("NPOL", "%u", &npol) != 1)
    throw invalid_argument ("NPOL did not exist in header");

  if (header.get ("NBIN", "%u", &nbin) != 1)
  {
    if (spip::Container::verbose)
      cerr << "spip::Container::read_header NBIN did not exist in header, assuming 1" << endl;
    nbin = 1;
  }

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
  {
    if (spip::Container::verbose)
      cerr << "spip::Container::read_header PICOSECONDS did not exist in header, assuming 0" << endl;
    utc_start_pico = 0;
  }

  if (header.get ("OBS_OFFSET", "%lu", &obs_offset) != 1)
    throw invalid_argument ("OBS_OFFSET did not exist in header");

  if (header.get ("OSAMP_NUMERATOR", "%u", &(oversampling_ratio[0])) != 1)
    oversampling_ratio[0] = 1;

  if (header.get ("OSAMP_DENOMINATOR", "%u", &(oversampling_ratio[1])) != 1)
    oversampling_ratio[1] = 1;
  
  if (header.get ("BYTES_PER_SECOND", "%lu", &bytes_per_second) != 1)
    throw invalid_argument ("BYTES_PER_SECOND did not exist in header");

  if (header.get ("FILE_SIZE", "%ld", &file_size) != 1)
    file_size = -1;
  else
  {
    seconds_per_file = double(file_size) / double(bytes_per_second);
    compute_file_size = true;
  }

  // update the calculated parameters
  recalculate();

  // assume little endian, unless specified
  if (header.get ("ENDIAN", "%s", &tmp_buf) == 1)
  {
    //cerr << "spip::Container::read_header ENDIAN=" << tmp_buf << endl;
    if (string(tmp_buf) == "LITTLE")
      endianness = spip::Endian::Little;
    else if (string(tmp_buf) == "BIG")
      endianness = spip::Endian::Big;
    else
      throw invalid_argument ("ENDIAN not LITTLE or BIG");
  }
  else
  {
    if (spip::Container::verbose)
      cerr << "spip::Container::read_header did not find ENDIAN keyword in header, assuming Little" << endl;
    endianness = spip::Endian::Little;
  }

  // assume Twos Complement Encoding, unless specified
  if (header.get ("ENCODING", "%s", &tmp_buf) == 1)
  {
    //cerr << "spip::Container::read_header ENCODING=" << tmp_buf << endl;
    if (string(tmp_buf) == "TWOSCOMPLEMENT")
      encoding = spip::Encoding::TwosComplement;
    else if (string(tmp_buf) == "OFFSETBINARY")
      encoding = spip::Encoding::OffsetBinary;
    else
      throw invalid_argument ("ENCODING not TWOSCOMPLEMENT or OFFSETBINARY");
  }
  else
  {
    if (spip::Container::verbose)
      cerr << "spip::Container::read_header did not find ENCODING keyword in header, assuming Twos Complement" << endl;
    encoding = spip::Encoding::TwosComplement;
  }

}

void spip::Container::write_header ()
{
  typedef std::numeric_limits< double > dbl;
  cerr.precision(dbl::max_digits10);

  if (header.set ("NANT", "%u", nsignal) < 0)
    throw invalid_argument ("Could not write NANT to header");

  if (header.set ("NCHAN", "%u", nchan) < 0)
    throw invalid_argument ("Could not write NCHAN to header");

  if (header.set ("NBIT", "%u", nbit) < 0)
    throw invalid_argument ("Could not write NBIT to header");

  if (header.set ("NPOL", "%u", npol) < 0)
    throw invalid_argument ("Could not write NPOL to header");

  if (header.set ("NBIN", "%u", nbin) < 0)
    throw invalid_argument ("Could not write NBIN to header");

  if (header.set ("NDIM", "%u", ndim) < 0)
    throw invalid_argument ("Could not write NDIM to header");

  if (header.set ("TSAMP", "%.16lf", tsamp) < 0)
    throw invalid_argument ("Could not write TSAMP to header");

  if (header.set ("BW", "%lf", bandwidth) < 0)
    throw invalid_argument ("Could not write BW to header");

  if (header.set ("FREQ", "%lf", centre_freq) < 0)
    throw invalid_argument ("Could not write FREQ to header");

  std::string utc_str = utc_start->get_gmtime();
  if (header.set ("UTC_START", "%s", utc_str.c_str()) < 0)
    throw invalid_argument ("Could not write UTC_START to header");

  if (header.set ("PICOSECONDS", "%lu", utc_start_pico) < 0)
    throw invalid_argument ("Could not write PICOSECONDS to (header");

  if (header.set ("OBS_OFFSET", "%lu", obs_offset) < 0)
    throw invalid_argument ("Could not write OBS_OFFSET to header");

  if (header.set ("OSAMP_NUMERATOR", "%u", (oversampling_ratio[0])) < 0)
    throw invalid_argument ("Could not write OSAMP_NUMERATOR to header");

  if (header.set ("OSAMP_DENOMINATOR", "%u", (oversampling_ratio[1])) < 0)
    throw invalid_argument ("Could not write OSAMP_DENOMINATOR to header");

  if (endianness == spip::Endian::Little)
  {
    if (header.set ("ENDIAN", "%s", "LITTLE") < 0)
      throw invalid_argument ("Could not write ENDIAN to header");
  }
  else
  {
    if (header.set ("ENDIAN", "%s", "BIG") < 0)
      throw invalid_argument ("Could not write ENDIAN to header");
  }

  // update the calculated parameters of the Container
  recalculate ();

  if (header.set ("BYTES_PER_SECOND", "%lu", bytes_per_second) < 0)
    throw invalid_argument ("Could not write BYTES_PER_SECOND to header");

  if (file_size != -1)
  {
    if (header.set ("FILE_SIZE", "%ld", file_size) < 0)
      throw invalid_argument ("Could not write FILE_SIZE to header");
  }

  if (header.set ("RESOLUTION", "%lu", resolution) < 0)
    throw invalid_argument ("Could not write RESOLUTION to header");
}

void spip::Container::clone_header (const spip::AsciiHeader &obj)
{
  header.clone (obj);
}

std::string spip::Container::get_order_string (spip::Ordering o)
{
  if (o == spip::Ordering::SFPT)
    return std::string("SFPT");
  if (o == spip::Ordering::TSPFB)
    return std::string("TSPFB");
  if (o == spip::Ordering::TFPS)
    return std::string("TFPS");
  if (o == spip::Ordering::TSPF)
    return std::string("TSPF");
  if (o == spip::Ordering::Custom)
    return std::string("Custom");
  return std::string("Unknown");
}

uint64_t spip::Container::calculate_bytes_per_second ()
{
  double nbit_per_samp = double(calculate_nbits_per_sample());
  //cerr << "spip::Container::calculate_bytes_per_second nbit_per_samp=" << nbit_per_samp << endl;
  double nsamp_per_second = double(1000000) / tsamp;
  //cerr << "spip::Container::calculate_bytes_per_second tsamp=" << tsamp << " nsamp_per_second=" << nsamp_per_second << endl;
  double nbit_per_second = nbit_per_samp * nsamp_per_second;
  //cerr << "spip::Container::calculate_bytes_per_second nbit_per_second=" << nbit_per_second<< endl;
  return uint64_t(nbit_per_second) / 8;
}

void spip::Container::recalculate ()
{
  // compute the new number of bits per sample
  bits_per_sample = calculate_nbits_per_sample();

  // compute the new bytes per second
  bytes_per_second = calculate_bytes_per_second ();

  // determine the resolution, based on the ordering
  if ((order == spip::Ordering::TFPS) || (order == spip::Ordering::TSPF) || (order == spip::Ordering::TSPFB))
    resolution = bits_per_sample / 8;
  else
    resolution = size;

  if (compute_file_size)
  {
    file_size = int64_t (seconds_per_file * bytes_per_second);
    if (verbose)
      cerr << "spip::Container::recalculate computed FILE_SIZE=" << file_size
           << " seconds_per_file=" << seconds_per_file 
           << " bytes_per_second=" << bytes_per_second << endl;
  }
}

void spip::Container::calculate_strides()
{
  if (spip::Container::verbose)
    cerr << "spip::Container::calculate_strides" << endl;

  if (order == spip::Ordering::SFPT)
  {
    bin_stride  = 0;
    dat_stride  = ndim;
    pol_stride  = ndat * dat_stride;
    chan_stride = npol * pol_stride;
    sig_stride  = nchan * chan_stride;
  }
  else if (order == spip::Ordering::TFPS)
  {
    bin_stride  = 0;
    sig_stride  = ndim;
    pol_stride  = nsignal * sig_stride;
    chan_stride = npol * pol_stride;
    dat_stride  = nchan * chan_stride;
  }
  else if (order == spip::Ordering::TSPF)
  {
    bin_stride  = 0;
    chan_stride = ndim;
    pol_stride  = nchan * chan_stride;
    sig_stride  = npol * pol_stride;
    dat_stride  = nsignal * sig_stride;
  }
  else if (order == spip::Ordering::TSPFB)
  {
    bin_stride  = ndim;
    chan_stride = nbin * bin_stride;
    pol_stride  = nchan * chan_stride;
    sig_stride  = npol * pol_stride;
    dat_stride  = nsignal * sig_stride; 
  }
  else if (order == spip::Ordering::Custom)
  {
    bin_stride  = 0;
    chan_stride = 0; 
    pol_stride  = 0;
    sig_stride  = 0;
    dat_stride  = 0;
  }
  else
    throw Error (InvalidState, "spip::Container::calculate_strides", "Unrecognized order: %s", get_order_string(order).c_str());

  if (spip::Container::verbose)
    cerr << "spip::Container::calculate_strides order=" << get_order_string(order) << " bin=" << bin_stride << " chan=" << chan_stride << " pol=" << pol_stride << " sig=" << sig_stride << " dat=" << dat_stride << endl;
}
