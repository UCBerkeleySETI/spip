/***************************************************************************
 *
 *   Copyright (C) 2018 Andrew Jameson
 *   Licensed under the Academic Free License version 2.1
 *
 ***************************************************************************/

#include "spip/Integration.h"

#include <stdexcept>
#include <cmath>

using namespace std;

spip::Integration::Integration () : Transformation<Container,Container>("Integration", outofplace)
{
  state = Signal::Intensity;

  dat_dec = 1;
  pol_dec = 1;
  chan_dec = 1;
  signal_dec = 1;

  buffer = NULL;
  buffer_idat = 0;
}

spip::Integration::~Integration ()
{
}

void spip::Integration::set_decimation(unsigned _dat_dec, unsigned _pol_dec, unsigned _chan_dec, unsigned _signal_dec)
{
  dat_dec = _dat_dec;
  pol_dec = _pol_dec;
  chan_dec = _chan_dec;
  signal_dec = _signal_dec;
}

//! intial configuration at the start of the data stream
void spip::Integration::configure (spip::Ordering output_order)
{
  ndat  = input->get_ndat ();
  nchan = input->get_nchan ();
  npol  = input->get_npol ();
  nbit  = input->get_nbit ();
  ndim  = input->get_ndim ();
  nsignal = input->get_nsignal ();

  if (ndim != 1)
    throw invalid_argument ("Integration::configure only ndim==1 supported");

  if (nbit != 32)
    throw invalid_argument ("Integration::configure nbit must be 32");

  if (nchan % chan_dec != 0)
    throw invalid_argument ("Integration::configure chan_dec must be a factor of nchan");

  if (pol_dec != 1 && npol != pol_dec)
    throw invalid_argument ("Integration::configure pol_dec must be equal to npol");

  if (signal_dec != 1 && nsignal != signal_dec)
    throw invalid_argument ("Integration::configure signal_dec must be equal to nsignal");

  bool valid_transform = false;
  if (input->get_order() == spip::Ordering::TSPF && output_order == spip::Ordering::TSPF)
    valid_transform = true;
  if (input->get_order() == spip::Ordering::TFPS && output_order == spip::Ordering::TFPS)
    valid_transform = true;
  if (!valid_transform)
    throw invalid_argument ("Integration::configure invalid ordering, must be TSPF->TSPF or TPFS->TPFS");

  // copy input header to output
  output->clone_header (input->get_header());

  // output will read the newly cloned header parameters
  output->read_header ();

  // update the parameters that this transformation will affect
  output->set_npol (npol / pol_dec);
  output->set_nchan (nchan / chan_dec);
  output->set_nsignal (nsignal / signal_dec);
  
  // the output file size should be a single sample
  unsigned bits_per_sample = output->calculate_nbits_per_sample();

  uint64_t file_size = bits_per_sample / 8;
  output->set_file_size (file_size);

  output->set_order (output_order);

  // update the output header parameters with the new details
  output->write_header ();
  
  // resize the output container
  prepare_output();

  // now prepare the buffer
  if (verbose)
    cerr << "spip::Integration::configure preparing buffer" << endl;
  buffer->clone_header(output->get_header());
  buffer->read_header();
  buffer->write_header();
  buffer->set_ndat (1);
  buffer->resize();
  buffer->zero();
}

void spip::Integration::prepare ()
{
  ndat  = input->get_ndat ();
  if (verbose)
    cerr << "spip::Integration::prepare ndat=" << ndat << endl;
}

//! simply copy input buffer to output buffer
void spip::Integration::transformation ()
{
  if (verbose)
    cerr << "spip::Integration::transformation()" << endl;

  // ensure output is appropriately sized
  prepare_output ();

  // apply data transformation
  if (input->get_order() == spip::Ordering::TSPF && output->get_order() == spip::Ordering::TSPF)
    transform_TSPF_to_TSPF();
  else if (input->get_order() == spip::Ordering::TFPS && output->get_order() == spip::Ordering::TFPS)
    transform_TFPS_to_TFPS();
  else
    throw invalid_argument ("spip::Integration::transformation unrecognized transformation state");
}

void spip::Integration::prepare_output ()
{
  uint64_t output_ndat = (buffer_idat + ndat) / dat_dec;
  if (verbose)
    cerr << "spip::Integration::prepare_output buffer_idat=" << buffer_idat 
         << " ndat=" << ndat << " dat_dec=" << dat_dec 
         << " output_ndat=" << output_ndat << endl;
  output->set_ndat (output_ndat);
  output->resize();
}

