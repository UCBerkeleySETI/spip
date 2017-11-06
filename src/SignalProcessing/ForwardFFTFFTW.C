/***************************************************************************
 *
 *   Copyright (C) 2016 Andrew Jameson
 *   Licensed under the Academic Free License version 2.1
 *
 ***************************************************************************/

#include "spip/ForwardFFTFFTW.h"

#include <stdexcept>
#include <cmath>

using namespace std;

spip::ForwardFFTFFTW::ForwardFFTFFTW ()
{
  plan = 0;
}

spip::ForwardFFTFFTW::~ForwardFFTFFTW ()
{
  if (plan)
    fftwf_destroy_plan (plan);
  plan = 0;
}

// configure the pipeline prior to runtime
void spip::ForwardFFTFFTW::configure ()
{
  if (nfft == 0)
    throw runtime_error ("ForwardFFTFFTW::configure nfft not set");

  spip::ForwardFFT::configure ();

  // build the FFT plan, with ordering SFPT -> TFPS
  configure_plan();
}

void spip::ForwardFFTFFTW::configure_plan ()
{
  if (verbose)
    cerr << "spip::ForwardFFTFFTW::configure_plan ndat=" << ndat << endl;

  // no function if ndat == 0
  if (ndat == 0)
    return;

  // if we are reconfiguring the batching, destroy the previous plan
  if (plan)
    fftwf_destroy_plan (plan);
  plan = 0;

  // build the FFT plan, with ordering SFPT -> TFPS
  int fftw_direction = FFTW_FORWARD;
  int fftw_flags = FFTW_ESTIMATE;
  
  fftwf_complex * in  = (fftwf_complex *) input->get_buffer();
  fftwf_complex * out = (fftwf_complex *) output->get_buffer();

  // configure the generic dimensions for the plan
  configure_plan_dimensions();

  plan = fftwf_plan_many_dft (rank, n, howmany,
                              in, inembed, istride, idist,
                              out, onembed, ostride, odist,
                              fftw_direction, fftw_flags); 
}

//! no special action required
void spip::ForwardFFTFFTW::prepare ()
{
  spip::ForwardFFT::prepare();
}

// convert to antenna minor order
void spip::ForwardFFTFFTW::transform_SFPT_to_TFPS ()
{
  if (verbose)
    cerr << "spip::ForwardFFTFFTW::transform_SFPT_to_TFPS()" << endl;

  fftwf_complex * in  = (fftwf_complex *) input->get_buffer();
  fftwf_complex * out = (fftwf_complex *) output->get_buffer();
  uint64_t out_offset;

  // iterate over input ordering of SFPT -> TFPS
  for (unsigned isig=0; isig<nsignal; isig++)
  {
    for (unsigned ichan=0; ichan<nchan; ichan++)
    {
      // output channel is offset by nfft
      unsigned ochan = ichan * nfft;
      for (unsigned ipol=0; ipol<npol; ipol++)
      {
        // process ndat samples via batched FFT
        out_offset = isig + (ipol * nsignal) + (ochan * npol * nsignal);
        fftwf_execute_dft (plan, in, out + out_offset);
        in += ndat;
      }
    }
  }
}

// convert to frequency minor order
void spip::ForwardFFTFFTW::transform_SFPT_to_TSPF ()
{
  if (verbose)
    cerr << "spip::ForwardFFTFFTW::transform_SFPT_to_TSPF()" << endl;

  fftwf_complex * in  = (fftwf_complex *) input->get_buffer();
  fftwf_complex * out = (fftwf_complex *) output->get_buffer();

  const uint64_t nchan_out = nchan * nfft;
  const uint64_t out_pol_stride = nchan_out;
  const uint64_t out_sig_stride = npol * out_pol_stride;

  // iterate over input ordering of SFPT -> TSPF
  for (unsigned isig=0; isig<nsignal; isig++)
  {
    const uint64_t out_sig_offset = isig * out_sig_stride;
    for (unsigned ichan=0; ichan<nchan; ichan++)
    {
      const uint64_t out_chan_offset = ichan * nfft;
      for (unsigned ipol=0; ipol<npol; ipol++)
      {
        const uint64_t out_pol_offset = ipol * out_pol_stride;

        // process ndat samples, in batches of nfft
        const uint64_t out_offset = out_sig_offset + out_chan_offset + out_pol_offset;

        fftwf_execute_dft (plan, in, out + out_offset);

        in += ndat;
      }
    }
  }
}

