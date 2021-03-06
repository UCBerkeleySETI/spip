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
  normalize = false;
  scale_fac = 1.0f;
}

spip::BackwardFFT::~BackwardFFT ()
{
}

void spip::BackwardFFT::set_nfft (int _nfft)
{
  if (_nfft <= 0)
    throw invalid_argument ("BackwardFFT::set_nfft nfft must be > 0");

  nfft = _nfft;
  if (normalize)
    scale_fac = 1.0f / float(nfft);
  else
    scale_fac = 1.0f;

  if (verbose)
    cerr << "spip::BackwardFFT::set_nfft nfft=" << nfft << " scale_fac=" << scale_fac << endl;

}

void spip::BackwardFFT::set_normalization (bool _normalize)
{ 
  normalize = _normalize;
  if (normalize)
  { 
    if (nfft > 0)
      scale_fac = 1.0f / float(nfft);
  }
  else
    scale_fac = 1.0f;
}

//! configure parameters at the start of a data stream
void spip::BackwardFFT::configure (spip::Ordering output_order)
{
  // this transformation requires the following parameters
  ndat  = input->get_ndat ();
  nbit = input->get_nbit ();
  ndim = input->get_ndim ();
  nchan = input->get_nchan ();
  npol  = input->get_npol ();
  nsignal = input->get_nsignal ();
  tsamp = input->get_tsamp();

  if (ndim != 2)
    throw invalid_argument ("BackwardFFT::configure input ndim != 2");

  if (nbit != 32)
    throw invalid_argument ("BackwardFFT::configure input nbit != 32");

  if (nchan % nfft != 0)
  {
    cerr << "spip::BackwardFFT::configure nchan=" << nchan << " nfft=" << nfft << endl;
    throw invalid_argument ("BackwardFFT::configure nchan must be divisible by nfft");
  }

  bool valid_transform = false;
  if (input->get_order() == TFPS && output_order == SFPT)
    valid_transform = true;
  if (input->get_order() == TSPF && output_order == SFPT)
    valid_transform = true;
  if (input->get_order() == SFPT && output_order == SFPT)
    valid_transform = true;

  if (!valid_transform)
    throw invalid_argument ("BackwardFFT::configure invalid ordering, allowed TFPS->SFPT, TSPF->SFPT or SFPT->SFPT");

  // copy input header to output
  output->clone_header (input->get_header());

  // output will read the newly cloned header parameters
  output->read_header ();

  // update the parameters that this transformation will affect
  output->set_nchan (nchan / nfft);
  output->set_tsamp (tsamp / nfft);
  output->set_order (spip::Ordering::SFPT);

  if (verbose)
  {
     cerr << "spip::BackwardFFT::configure nchan: " << nchan 
          << " -> " << nchan / nfft << endl;
     cerr << "spip::BackwardFFT::configure tsamp: " << tsamp 
          << " -> " << tsamp / nfft << endl;
  }

  // update the output header parameters with the new details
  output->write_header ();

  // ensure the output is appropriately sized
  prepare_output();

  // TODO change the UTC_START to be related to the new time resolution
}

void spip::BackwardFFT::configure_plan_dimensions ()
{
  // batch over multiple time samples for a given output channel, antenna and pol
  // each batch produces nfft out time sample, from nfft input channels 
  nbatch = ndat;
  nchan_out = nchan / nfft;

  if (verbose)
    cerr << "spip::BackwardFFTFFTW::configure_plan_dimensions nbatch=" << nbatch << " nfft=" << nfft << endl;

  rank = 1;                       // 1D transform
  n[0] = nfft;                    // of length nfft
  howmany = nbatch;

  inembed[0] = nfft;
  onembed[0] = nfft;

  if ((input->get_order() == TFPS) && (output->get_order() == SFPT))
  {
    istride = npol * nsignal;     // stride between samples
    idist = nchan * istride;      // stride between FFT blocks
    ostride = 1;                  // stride between samples
    odist = nfft * ostride;       // stride between FFT blacks
  }
  else if ((input->get_order() == TSPF) && (output->get_order() == SFPT))
  {
    istride = npol * nsignal;     // stride between samples
    idist = nchan * istride;      // stride between FFT blocks
    ostride = 1;                  // stride between samples
    odist = nfft * ostride;       // stride between FFT blacks
  }
  else if ((input->get_order() == SFPT) && (output->get_order() == SFPT))
  {
    istride = npol * nbatch;      // stride between channels
    idist = 1;                    // stride between FFT blocks
    ostride = 1;                  // stride between samples
    odist = nfft * ostride;       // stride between FFT blocks
  }
  else
    throw invalid_argument ("BackwardFFT::configure_plan_dimensions unsupported input/output ordering");

  if (verbose)
    cerr << "spip::BackwardFFT::configure_plan_dimensions istride=" << istride 
         << " idist=" << idist << " ostride=" << ostride << " odist=" << odist << endl;

}

//! prepare prior to each transformation call
void spip::BackwardFFT::prepare ()
{
}

//! ensure meta-data is correct in output
void spip::BackwardFFT::prepare_output ()
{
  // update the output parameters that may change from block to block
  output->set_ndat (ndat * nfft);

  // resize output based on configuration
  output->resize();
}

//! simply copy input buffer to output buffer
void spip::BackwardFFT::transformation ()
{
  if (verbose)
    cerr << "void spip::BackwardFFT::transformation()" << endl;

  if (nchan % nfft != 0)
    throw invalid_argument ("BackwardFFT::transformation nchan must be divisible by nfft");

  // ensure the output is appropriately sized
  prepare_output();

  if (ndat == 0)
  {
    cerr << "spip::BackwardFFT::transformation ndat==0, ignoring" << endl;
    return;
  }
    
  // apply data transformation
  if ((input->get_order() == TFPS) && (output->get_order() == SFPT))
  {
    transform_TFPS_to_SFPT ();
  }
  else if ((input->get_order() == TSPF) && (output->get_order() == SFPT))
  {
    transform_TSPF_to_SFPT ();
  }
  else if ((input->get_order() == SFPT) && (output->get_order() == SFPT))
  {
    transform_SFPT_to_SFPT ();
  }
  else
  {
    throw runtime_error ("BackwardFFT::transform unsupport input to output conversion");
  }

  if (normalize)
  {
    if (verbose)
      cerr << "spip::BackwardFFT::transform normalize_output()" << endl;
    normalize_output();
  }

}

