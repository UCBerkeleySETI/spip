/***************************************************************************
 *
 *   Copyright (C) 2018 Andrew Jameson
 *   Licensed under the Academic Free License version 2.1
 *
 ***************************************************************************/

#include "spip/SampleFold.h"
#include "spip/Error.h"

#include <stdexcept>
#include <cmath>

using namespace std;

spip::SampleFold::SampleFold () : Transformation<Container,Container>("SampleFold", outofplace)
{
  nfolding_bin = 0;
  dat_offset = 0;
  dat_dec = 0;
  buffer_idat = 0;
}

spip::SampleFold::~SampleFold ()
{
}

//! set folding parameters, assuming 1 sample per bin
void spip::SampleFold::set_periodicity (unsigned _nbin, uint64_t _dat_offset, uint64_t _dat_dec)
{
  if (verbose)
    cerr << "spip::SampleFold::set_periodicity nfolding_bin=" << _nbin << " dat_dec=" << _dat_dec << endl;
  // number of samples/bins to fold into
  nfolding_bin = _nbin;
  dat_offset = dat_offset;
  dat_dec = _dat_dec;
}

//! intial configuration at the start of the data stream
void spip::SampleFold::configure (spip::Ordering output_order)
{
  if (verbose)
    cerr << "spip::SampleFold::configure()" << endl;

  ndat  = input->get_ndat ();
  nchan = input->get_nchan ();
  npol  = input->get_npol ();
  nbit  = input->get_nbit ();
  ndim  = input->get_ndim ();
  nbin  = input->get_nbin ();
  nsignal = input->get_nsignal ();
  tsamp = input->get_tsamp();

  if (nbit != 32)
    throw Error (InvalidState, "spip::SampleFold::configure", "input nbit [%u] must be 32", nbit);

  if (nbin != 1)
    throw Error (InvalidState, "spip::SampleFold::configure", "input nbin [%u] must be 1", nbin);

  bool valid_transform = false;
  if (input->get_order() == spip::Ordering::SFPT && output_order == spip::Ordering::TSPFB)
    valid_transform = true;
  if (input->get_order() == spip::Ordering::TFPS && output_order == spip::Ordering::TSPFB)
    valid_transform = true;
  if (input->get_order() == spip::Ordering::TSPF && output_order == spip::Ordering::TSPFB)
    valid_transform = true;
  if (!valid_transform)
    throw invalid_argument ("SampleFold::configure invalid ordering, input must be SFPT, TFPS or TSPF and output TSPFB");

  if (verbose)
    cerr << "spip::SampleFold::configure configuring output" << endl;

  // copy input header to output
  output->clone_header (input->get_header());

  // output will read the newly cloned header parameters
  output->read_header ();

  // update the parameters that this transformation will affect
  output->set_nbin (nfolding_bin);
  output->set_tsamp (tsamp * dat_dec);
  output->set_order (output_order);

  // the output file size should be a single sample
  unsigned bits_per_sample = output->calculate_nbits_per_sample();
  uint64_t file_size = bits_per_sample / 8;
  output->set_file_size (file_size);

  // update the output header parameters with the new details
  output->write_header ();
  
  if (verbose)
    cerr << "spip::SampleFold::configure prepare_output" << endl;

  if (verbose)
    cerr << "spip::SampleFold::set_periodicity output nbin=" << output->get_nbin() << endl;

  // resize the output container
  prepare_output();

  if (verbose)
    cerr << "spip::SampleFold::configure configuring buffer" << endl;

  // build the buffer and ensure it contains 1 time sample for integration
  buffer->clone_header(output->get_header());
  buffer->read_header();
  buffer->write_header();
  buffer->set_ndat (1);
  buffer->set_order (output_order);
  buffer->resize();
  buffer->zero();
  if (verbose)
    cerr << "spip::SampleFold::configure done" << endl;

}

//! read the number in input data points
void spip::SampleFold::prepare ()
{
  ndat = input->get_ndat ();
  if (verbose)
    cerr << "spip::SampleFold::prepare ndat=" << ndat << endl;
}

//! simply copy input buffer to output buffer
void spip::SampleFold::transformation ()
{
  if (verbose)
    cerr << "spip::SampleFold::transformation()" << endl;

  // ensure output is appropriately sized
  prepare_output ();

  // apply data transformation
  if (input->get_order() == spip::Ordering::SFPT && output->get_order() == spip::Ordering::TSPFB)
    transform_SFPT_to_TSPFB();
  else if (input->get_order() == spip::Ordering::TSPF && output->get_order() == spip::Ordering::TSPFB)
    transform_TSPF_to_TSPFB();
  else if (input->get_order() == spip::Ordering::TFPS && output->get_order() == spip::Ordering::TSPFB)
    transform_TFPS_to_TSPFB();
  else
    throw Error(InvalidState, "spip::SampleFold::transformation", "unrecognized transformation state");
}

void spip::SampleFold::prepare_output ()
{
  // the number of samples buffered after this operation
  uint64_t buffered_ndat = buffer_idat + ndat;

  // the number of output samples to be produced, expect 0 or a few 
  uint64_t output_ndat = buffered_ndat / dat_dec;

  if (verbose)
    cerr << "spip::SampleFold::prepare_output buffer_idat=" << buffer_idat
         << " ndat=" << ndat << " dat_dec=" << dat_dec
         << " output_ndat=" << output_ndat << endl;
  output->set_ndat (output_ndat);

  if (verbose)
    cerr << "spip::SampleFold::prepare_output output->resize()" << endl;
  output->resize();
}
