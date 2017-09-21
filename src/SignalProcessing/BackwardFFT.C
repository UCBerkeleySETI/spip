/***************************************************************************
 *
 *   Copyright (C) 2016 Andrew Jameson
 *   Licensed under the Academic Free License version 2.1
 *
 ***************************************************************************/

#include "spip/BackwardFFT.h"

#include <stdexcept>
#include <cmath>

using namespace std;

spip::BackwardFFT::BackwardFFT () : Transformation<Container,Container>("BackwardFFT", outofplace)
{
  nfft = 0;
}

spip::BackwardFFT::~BackwardFFT ()
{
}

void spip::BackwardFFT::set_nfft (int _nfft)
{
  nfft = _nfft;
}

//! configure parameters at the start of a data stream
void spip::BackwardFFT::configure ()
{
  // this transformation requires the following parameters
  ndat  = input->get_ndat ();
  nbit = input->get_nbit ();
  ndim = input->get_ndim ();
  nchan = input->get_nchan ();
  npol  = input->get_npol ();
  nsignal = input->get_nsignal ();
  tsamp = input->get_tsamp();

  cerr << "spip::BackwardFFT::configure ndat=" << ndat << " nfft=" << nfft << endl;

  if (ndim != 2)
    throw invalid_argument ("BackwardFFT::configure input ndim != 2");

  if (nbit != 32)
    throw invalid_argument ("BackwardFFT::configure input nbit != 32");

  if (ndat % nfft != 0)
  {
    cerr << "spip::BackwardFFT::configure ndat=" << ndat << " nfft=" << nfft << endl;
    throw invalid_argument ("BackwardFFT::configure ndat must be divisible by nfft");
  }

  if (input->get_order() != spip::Ordering::SFPT)
    throw invalid_argument ("BackwardFFT::configure input order must be SFPT");

  // copy input header to output
  output->clone_header (input->get_header());

  // output will read the newly cloned header parameters
  output->read_header ();

  // update the parameters that this transformation will affect
  output->set_nchan (nchan * nfft);
  output->set_tsamp (tsamp / nfft);
  output->set_order (spip::Ordering::TFPS);

  if (verbose)
  {
     cerr << "spip::BackwardFFT::configure nchan: " << nchan 
          << " -> " << nchan * nfft << endl;
     cerr << "spip::BackwardFFT::configure tsamp: " << tsamp 
          << " -> " << tsamp / nfft << endl;
  }

  // update the output header parameters with the new details
  output->write_header ();

  // ensure the output is appropriately sized
  prepare_output();

  // TODO change the UTC_START to be related to the new time resolution
}

//! prepare prior to each transformation call
void spip::BackwardFFT::prepare ()
{
  ndat = input->get_ndat();
  uint64_t remainder = ndat % nfft;
  if (remainder != 0)
  {
    if (verbose)
      cerr << "spip::BackwardFFT::prepare truncating ndat from " << ndat << " to " << ndat - remainder << endl;
    ndat -= remainder;
  }
}

//! ensure meta-data is correct in output
void spip::BackwardFFT::prepare_output ()
{
  // update the output parameters that may change from block to block
  output->set_ndat (ndat / nfft);

  // resize output based on configuration
  output->resize();
}

//! simply copy input buffer to output buffer
void spip::BackwardFFT::transformation ()
{
  if (ndat % nfft != 0)
    throw invalid_argument ("BackwardFFT::transformation ndat must be divisible by nfft");

  // ensure the output is appropriately sized
  prepare_output();

  if (ndat == 0)
  {
    cerr << "spip::BackwardFFT::transformation ndat==0, ignoring" << endl;
    return;
  }
    
  // apply data transformation
  transform ();
}

