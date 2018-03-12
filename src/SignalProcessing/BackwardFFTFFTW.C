/***************************************************************************
 *
 *   Copyright (C) 2016 Andrew Jameson
 *   Licensed under the Academic Free License version 2.1
 *
 ***************************************************************************/

#include "spip/BackwardFFTFFTW.h"

#include <stdexcept>
#include <cmath>

using namespace std;

spip::BackwardFFTFFTW::BackwardFFTFFTW ()
{
  plan = 0;
}

spip::BackwardFFTFFTW::~BackwardFFTFFTW ()
{
  if (plan)
    fftwf_destroy_plan (plan);
  plan = 0;
}

void spip::BackwardFFTFFTW::configure (spip::Ordering output_order)
{
  if (nfft == 0)
    throw runtime_error ("BackwardFFTFFTW::configure nfft not set");

  spip::BackwardFFT::configure (output_order);

  configure_plan ();
}

void spip::BackwardFFTFFTW::configure_plan ()
{
  if (verbose)
    cerr << "spip::BackwardFFTFFTW::configure_plan ndat=" << ndat << endl;

  // no function if ndat == 0
  if (ndat == 0)
    return;

  // if we are reconfiguring the batching, destroy the previous plan
  if (plan)
    fftwf_destroy_plan (plan);
  plan = 0;

  // configure the generic FFT plan parameters
  configure_plan_dimensions ();

  // build the FFT plan, with ordering TFPS -> SFPT
  int fftw_direction = FFTW_BACKWARD;
  int fftw_flags = FFTW_ESTIMATE;

  fftwf_complex * in  = (fftwf_complex *) input->get_buffer();
  fftwf_complex * out = (fftwf_complex *) output->get_buffer();

  plan = fftwf_plan_many_dft (rank, n, howmany,
                              in, inembed, istride, idist,
                              out, onembed, ostride, odist,
                              fftw_direction, fftw_flags);
}


//! no special action required
void spip::BackwardFFTFFTW::prepare ()
{
  spip::BackwardFFT::prepare();
}

//! perform Backward FFT using FFTW
void spip::BackwardFFTFFTW::transform_TFPS_to_SFPT ()
{
  if (verbose)
    cerr << "spip::BackwardFFTFFTW::transform_TFPS_to_SFPT()" << endl;
  fftwf_complex * in  = (fftwf_complex *) input->get_buffer();
  fftwf_complex * out = (fftwf_complex *) output->get_buffer();

  // this FFT will produce the specified number of output channels
  const uint64_t ndat_out = ndat * nfft;
  const uint64_t istride = nsignal * npol * nfft;
  const uint64_t ostride = npol * ndat_out;

  for (uint64_t idat=0; idat<ndat; idat++)
  {
    for (unsigned ochan=0; ochan<nchan_out; ochan++)
    {
      unsigned ipolsig = 0;
      unsigned opolsig = 0;
      for (unsigned ipol=0; ipol<npol; ipol++)
      {
        for (unsigned isig=0; isig<nsignal; isig++)
        {
          fftwf_execute_dft (plan, in + ipolsig, out);
          ipolsig += 1;
          opolsig += ndat_out;
        }
      }
      in += istride;
      out += ostride;
    }
  }
}

void spip::BackwardFFTFFTW::transform_TSPF_to_SFPT ()
{
  if (verbose)
    cerr << "spip::BackwardFFTFFTW::transform_TSPF_to_SFPT()" << endl;
  fftwf_complex * in  = (fftwf_complex *) input->get_buffer();
  fftwf_complex * out = (fftwf_complex *) output->get_buffer();

  const uint64_t in_chan_stride = nchan;    
  const uint64_t ndat_out = ndat * nfft;
  const uint64_t out_pol_stride = ndat_out;
  const uint64_t out_chan_stride = npol * out_pol_stride;
  const uint64_t out_sig_stride = nchan_out * out_chan_stride;

  // FFT batch is for input ndat
  for (unsigned isig=0; isig<nsignal; isig++)
  {
    const uint64_t out_sig_offset = isig * out_sig_stride;
    for (unsigned ipol=0; ipol<npol; ipol++)
    {
      const uint64_t out_pol_offset = ipol * out_pol_stride;
      for (unsigned ochan=0; ochan<nchan_out; ochan++)
      {
        const uint64_t out_chan_offset = ochan * out_chan_stride;
        const uint64_t out_offset = out_chan_offset + out_sig_offset + out_pol_offset;
#ifdef _DEBUG
        cerr << "ochan=" << ochan << " isig=" << isig << " ipol=" << ipol << " out_chan_offset=" 
             << out_chan_offset << " out_sig_offset=" << out_sig_offset << " out_pol_offset=" 
             << out_pol_offset << " out_offset=" << out_offset << endl;
#endif
        fftwf_execute_dft (plan, in, out + out_offset);
        in += in_chan_stride;
      }
    }
  }
}

//TODO fix this
void spip::BackwardFFTFFTW::transform_SFPT_to_SFPT ()
{
  if (verbose)
    cerr << "spip::BackwardFFTFFTW::transform_SFPT_to_SFPT()" << endl;
  fftwf_complex * in  = (fftwf_complex *) input->get_buffer();
  fftwf_complex * out = (fftwf_complex *) output->get_buffer();

  const uint64_t in_chan_stride = nchan;
  const uint64_t ndat_out = ndat * nfft;
  const uint64_t out_pol_stride = ndat_out;
  const uint64_t out_chan_stride = npol * out_pol_stride;
  const uint64_t out_sig_stride = nchan_out * out_chan_stride;

  // FFT batch is for input ndat
  for (unsigned isig=0; isig<nsignal; isig++)
  {
    const uint64_t out_sig_offset = isig * out_sig_stride;
    for (unsigned ipol=0; ipol<npol; ipol++)
    {
      const uint64_t out_pol_offset = ipol * out_pol_stride;
      for (unsigned ochan=0; ochan<nchan_out; ochan++)
      {
        const uint64_t out_chan_offset = ochan * out_chan_stride;
        const uint64_t out_offset = out_chan_offset + out_sig_offset + out_pol_offset;
#ifdef _DEBUG
        cerr << "ochan=" << ochan << " isig=" << isig << " ipol=" << ipol << " out_chan_offset="
             << out_chan_offset << " out_sig_offset=" << out_sig_offset << " out_pol_offset="
             << out_pol_offset << " out_offset=" << out_offset << endl;
#endif
        fftwf_execute_dft (plan, in, out + out_offset);
        in += in_chan_stride;
      }
    }
  }
}

void spip::BackwardFFTFFTW::normalize_output ()
{
  if (verbose)
    cerr << "spip::BackwardFFTFFTW::transform_normalize_output()" << endl;

  fftwf_complex * out = (fftwf_complex *) output->get_buffer();
  uint64_t nval = ndat * nsignal * nchan * npol;

  if (verbose)
    cerr << "spip::BackwardFFTFFTW::transform_normalize_output nval=" << nval 
         << " scale_fac=" << scale_fac << endl;
  for (uint64_t ival=0; ival<nval; ival++)
  {
    out[ival][0] = out[ival][0] * scale_fac;
    out[ival][1] = out[ival][1] * scale_fac;
  }
}



