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
  normalize = true;
  apply_fft_shift = false;
  scale_fac = 1.0f;
  conditioned = NULL;
}

spip::ForwardFFT::~ForwardFFT ()
{
}

void spip::ForwardFFT::set_nfft (int _nfft)
{
  if (_nfft <= 0)
    throw invalid_argument ("ForwardFFT::set_nfft nfft must be > 0");

  nfft = _nfft;
  if (normalize)
    scale_fac = 1.0f / float(nfft);
  else
    scale_fac = 1.0f;

  if (verbose)
    cerr << "spip::ForwardFFT::set_nfft nfft=" << nfft << " scale_fac=" << scale_fac << endl;
}

void spip::ForwardFFT::set_normalization (bool _normalize)
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
void spip::ForwardFFT::configure (spip::Ordering output_order)
{
  // this transformation requires the following parameters
  ndat  = input->get_ndat ();
  nbit = input->get_nbit ();
  ndim = input->get_ndim ();
  nchan = input->get_nchan ();
  npol  = input->get_npol ();
  nsignal = input->get_nsignal ();
  tsamp = input->get_tsamp ();
  int dsb = input->get_dual_sideband ();

  if (verbose)
    cerr << "spip::ForwardFFT::configure ndat=" << ndat << " nfft=" << nfft 
         << " nchan=" << nchan << " npol=" << npol << " dsb=" << dsb << endl;

  if (ndim != 2)
    throw invalid_argument ("ForwardFFT::configure input ndim != 2");

  if (nbit != 32)
    throw invalid_argument ("ForwardFFT::configure input nbit != 32");

  if (nchan != 1)
    throw invalid_argument ("ForwardFFT::configure input nchan != 1");

  if (ndat % nfft != 0)
  {
    cerr << "spip::ForwardFFT::configure ndat=" << ndat << " nfft=" << nfft << endl;
    throw invalid_argument ("ForwardFFT::configure ndat must be divisible by nfft");
  }

  bool valid_transform = false;
  if (input->get_order() == SFPT && output_order == TFPS)
    valid_transform = true;
  if (input->get_order() == SFPT && output_order == TSPF)
    valid_transform = true;
  if (input->get_order() == SFPT && output_order == SFPT)
    valid_transform = true;

  if (!valid_transform)
    throw Error(InvalidState, "spip::ForwardFFT::configure", 
                "invalid ordering, allowed SFPT->TFPS, SFPT->TSPF, SFPT->SFPT");

  // apply FFT shift if the data are dual sideband
  if (dsb == 1)
    apply_fft_shift = true;

  // if we need to shift or scale the data prior to Forward FFT
  if ((normalize) || (apply_fft_shift))
  {
    // copy input header to output
    conditioned->clone_header (input->get_header());

    // read newly cloned header params
    conditioned->read_header();

    conditioned->set_dual_sideband (0);
    conditioned->set_order (input->get_order());
    conditioned->write_header();

    output->clone_header (conditioned->get_header());
  }
  else
    output->clone_header (input->get_header());

  // output will read the newly cloned header parameters
  output->read_header ();

  // update the parameters that this transformation will affect
  output->set_nchan (nchan * nfft);
  output->set_tsamp (tsamp * nfft);
  output->set_order (output_order);

  if (verbose)
  {
     cerr << "spip::ForwardFFT::configure nchan: " << nchan 
          << " -> " << nchan * nfft << endl;
     cerr << "spip::ForwardFFT::configure tsamp: " << tsamp 
          << " -> " << tsamp * nfft << endl;
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
  howmany = nbatch;         // number of fft batches to perform

  inembed[0] = nfft;
  onembed[0] = nfft;

  if (verbose)
    cerr << "spip::ForwardFFT::configure_plan_dimensions ndat=" << ndat  << " nfft=" << nfft << " nbatch=" << nbatch << endl;

  if ((input->get_order() == SFPT) && (output->get_order() == TFPS))
  {
    istride = 1;                  // stride between samples
    idist = nfft * istride;       // stride between FFT blocks
    ostride = npol * nsignal;     // stride between channels
    odist = nchan_out * ostride;  // stride between FFT blocks
  }
  else if ((input->get_order() == SFPT) && (output->get_order() == TSPF))
  {
    istride = 1;                  // stride between samples
    idist = nfft * istride;       // stride between FFT blocks
    ostride = 1;                  // stride between channels
    odist = nchan_out * nsignal * npol;  // stride between FFT blocks
  }
  else if ((input->get_order() == SFPT) && (output->get_order() == SFPT))
  {
    istride = 1;                  // stride between samples
    idist = nfft * istride;       // stride between FFT blocks
    ostride = npol * nbatch;      // stride between channels
    odist = 1;                    // stride between FFT blocks
  }
  else
  {
    throw invalid_argument ("ForwardFFT::configure_plan_dimensions input/output order not supported");
  }
  if (verbose)
    cerr << "spip::ForwardFFT::configure_plan_dimensions istride=" << istride << " idist=" << idist << " ostride=" << ostride << " odist=" << odist << endl;
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
  if ((normalize) || (apply_fft_shift))
  {
    conditioned->set_ndat (ndat);
    conditioned->resize();
  }

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
    
  if (apply_fft_shift || normalize)
  {
    if (verbose)
      cerr << "spip::ForwardFFT::transform condition()" << endl;
    // condition the data
    condition ();
  }

/*
  if (apply_fft_shift)
  {
    if (verbose)
      cerr << "spip::ForwardFFT::transform fft_shift()" << endl;
    fft_shift();
  }
*/

  // apply data transformation
  if ((input->get_order() == SFPT) && (output->get_order() == TFPS))
  {
    if (verbose)
      cerr << "spip::ForwardFFT::transform transform_SFPT_to_TFPS()" << endl;
    transform_SFPT_to_TFPS ();
  }
  else if ((input->get_order() == SFPT) && (output->get_order() == TSPF))
  {
    if (verbose)
      cerr << "spip::ForwardFFT::transform transform_SFPT_to_TSPF()" << endl;
    transform_SFPT_to_TSPF ();
  }
  else if ((input->get_order() == SFPT) && (output->get_order() == SFPT))
  {
    if (verbose)
      cerr << "spip::ForwardFFT::transform transform_SFPT_to_SFPT()" << endl;
    transform_SFPT_to_SFPT ();
  }
  else
  {
    throw runtime_error ("ForwardFFT::transform unsupport input to output conversion");
  }

/*
  if (normalize)
  {
    if (verbose)
      cerr << "spip::ForwardFFT::transform normalize_output()" << endl;
    normalize_output();
  }
*/
}

