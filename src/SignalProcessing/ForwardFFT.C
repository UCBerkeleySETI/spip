/***************************************************************************
 *
 *   Copyright (C) 2016 Andrew Jameson
 *   Licensed under the Academic Free License version 2.1
 *
 ***************************************************************************/

#include "spip/ForwardFFT.h"

#include <stdexcept>
#include <cmath>

using namespace std;

spip::ForwardFFT::ForwardFFT () : Transformation<Container,Container>("ForwardFFT", outofplace)
{
  nfft = 0;
  nbatch = 0;
}

spip::ForwardFFT::~ForwardFFT ()
{
}

void spip::ForwardFFT::set_nfft (int _nfft)
{
  nfft = _nfft;
}

//! configure parameters at the start of a data stream
void spip::ForwardFFT::configure ()
{
  // this transformation requires the following parameters
  ndat  = input->get_ndat ();
  nbit = input->get_nbit ();
  ndim = input->get_ndim ();
  nchan = input->get_nchan ();
  npol  = input->get_npol ();
  nsignal = input->get_nsignal ();
  tsamp = input->get_tsamp();

  cerr << "spip::ForwardFFT::configure ndat=" << ndat << " nfft=" << nfft << endl;

  if (ndim != 2)
    throw invalid_argument ("ForwardFFT::configure input ndim != 2");

  if (nbit != 32)
    throw invalid_argument ("ForwardFFT::configure input nbit != 32");

  if (ndat % nfft != 0)
  {
    cerr << "spip::ForwardFFT::configure ndat=" << ndat << " nfft=" << nfft << endl;
    throw invalid_argument ("ForwardFFT::configure ndat must be divisible by nfft");
  }

  if (input->get_order() != spip::Ordering::SFPT)
    throw invalid_argument ("ForwardFFT::configure input order must be SFPT");

  // copy input header to output
  output->clone_header (input->get_header());

  // output will read the newly cloned header parameters
  output->read_header ();

  // update the parameters that this transformation will affect
  output->set_nchan (nchan * nfft);
  output->set_tsamp (tsamp / nfft);
  output->set_order (spip::Ordering::TSPF);

  if (verbose)
  {
     cerr << "spip::ForwardFFT::configure nchan: " << nchan 
          << " -> " << nchan * nfft << endl;
     cerr << "spip::ForwardFFT::configure tsamp: " << tsamp 
          << " -> " << tsamp / nfft << endl;
  }

  // update the output header parameters with the new details
  output->write_header ();

  // ensure the output is appropriately sized
  prepare_output();

  // TODO change the UTC_START to be related to the new time resolution
}

void spip::ForwardFFT::configure_plan_dimensions()
{
  // batch over multiple time samples for a given input channel, antenna and pol
  // each batch produces 1 out time sample, nfft channels 
  nbatch = ndat / nfft;
  nchan_out = nchan * nfft;

  rank = 1;                 // 1D transform
  n[0] = nfft;              // of length nfft
  howmany = nbatch;

  inembed[0] = nfft;
  onembed[0] = nfft;

  if (verbose)
    cerr << "spip::ForwardFFT::configure_plan_dimensions ndat=" << ndat  << " nfft=" << nfft << " nbatch=" << nbatch << endl;

  if ((input->get_order() == SFPT) && (output->get_order() == TFPS))
  {
    istride = 1 ;                 // stride between samples
    idist = nfft * istride;       // stride between FFT blocks
    ostride = npol * nsignal;     // stride between samples
    odist = nchan_out * ostride;  // stride between FFT blocks
  }
  else if ((input->get_order() == SFPT) && (output->get_order() == TSPF))
  { 
    istride = 1;                 // stride between samples
    idist = nfft * istride;       // stride between FFT blocks
    ostride = npol * nsignal;     // stride between samples
    odist = nchan_out * ostride;  // stride between FFT blocks
  }
  else
  {
    throw invalid_argument ("ForwardFFT::configure_plan_dimensions input/output order not supported");
  }
}

//! prepare prior to each transformation call
void spip::ForwardFFT::prepare ()
{
  ndat = input->get_ndat();

  // check that ndat is a multiple of nfft
  uint64_t remainder = ndat % nfft;
  if (remainder != 0)
  {
    if (verbose)
      cerr << "spip::ForwardFFT::prepare truncating ndat from " << ndat << " to " << ndat - remainder << endl;
    ndat -= remainder;
  }

  // check that the batching length is correct
  if (nbatch != ndat / nfft)
  {
    if (verbose)
      cerr << "spip::ForwardFFT::prepare reconfiguring FFT plan" << endl;
    configure_plan ();
  }
}

//! ensure meta-data is correct in output
void spip::ForwardFFT::prepare_output ()
{
  // update the output parameters that may change from block to block
  output->set_ndat (ndat / nfft);

  // resize output based on configuration
  output->resize();
}

//! simply copy input buffer to output buffer
void spip::ForwardFFT::transformation ()
{
  if (ndat % nfft != 0)
    throw invalid_argument ("ForwardFFT::transformation ndat must be divisible by nfft");

  // ensure the output is appropriately sized
  prepare_output();

  if (ndat == 0)
  {
    cerr << "spip::ForwardFFT::transformation ndat==0, ignoring" << endl;
    return;
  }

  if (verbose)
    cerr << "spip::ForwardFFT::transform nfft=" << nfft << endl;
    
  // apply data transformation
  if ((input->get_order() == SFPT) && (output->get_order() == TFPS))
  {
    transform_SFPT_to_TFPS ();
  }
  else if ((input->get_order() == SFPT) && (output->get_order() == TSPF))
  {
    transform_SFPT_to_TSPF ();
  }
  else
  {
    throw runtime_error ("ForwardFFT::transform unsupport input to output conversion");
  }
}

