/***************************************************************************
 *
 *   Copyright (C) 2016 Andrew Jameson
 *   Licensed under the Academic Free License version 2.1
 *
 ***************************************************************************/

#include "spip/BatchedBackwardFFT.h"

#include <stdexcept>
#include <cmath>

using namespace std;

spip::BatchedBackwardFFT::BatchedBackwardFFT () : Transformation<Container,Container>("BatchedBackwardFFT", outofplace)
{
  nfft = 0;
}

spip::BatchedBackwardFFT::~BatchedBackwardFFT ()
{
}

void spip::BatchedBackwardFFT::configure ()
{
  ndat  = input->get_ndat ();
  ndim  = input->get_ndim ();
  nchan = input->get_nchan ();
  npol  = input->get_npol ();
  nsignal = input->get_nsignal ();
  tsamp = input->get_tsamp ();

  nfft = nfft;
  nchan_out = nchan / nfft;
  tsamp_out = tsamp * nfft;

  if (ndim != 2)
    throw invalid_argument ("BatchedBackwardFFT::configure only ndim==2 supported");

  if (nchan % nfft != 0)
    throw invalid_argument ("BatchedBackwardFFT::configure ndat must be divisible by nfft");

  if (input->get_order() != TFPS)
    throw invalid_argument ("BatchedBackwardFFT::configure input was not TFPS order");

  // copy input header to output
  output->clone_header (input->get_header());

  // output will read the newly cloned header parameters
  output->read_header ();

  // update the parameters that this transformation will affect
  output->set_nchan (nchan_out);
  output->set_tsamp (tsamp_out);
  output->set_order (spip::Ordering::TFPS);

  // update the output header parameters with the new details
  output->write_header ();

  // ensure output is configured
  prepare_output();
}

void spip::BatchedBackwardFFT::prepare ()
{
  ndat  = input->get_ndat ();
}

//! prepare the output data product
void spip::BatchedBackwardFFT::prepare_output ()
{
  // update the number of output samples
  output->set_ndat (ndat * nfft);

  // resize output based on configuration
  output->resize();
}

//! simply copy input buffer to output buffer
void spip::BatchedBackwardFFT::transformation ()
{
  if (nchan % nfft != 0)
    throw invalid_argument ("BatchedBackwardFFT::transformation nchan must be divisible by nfft");

  //! adjust output size
  prepare_output ();

  if (ndat == 0)
  {
    cerr << "spip::BatchedBackwardFFT::transformation ndat==0, ignoring" << endl;
    return;
  }


  // apply data transformation
  transform ();
}

