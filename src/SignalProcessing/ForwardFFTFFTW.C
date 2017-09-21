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

void spip::ForwardFFTFFTW::configure ()
{
  if (nfft == 0)
    throw runtime_error ("ForwardFFTFFTW::configure nfft not set");

  spip::ForwardFFT::configure ();

  int fftw_direction = FFTW_FORWARD;
  int fftw_flags = FFTW_ESTIMATE;

  fftwf_complex * in  = (fftwf_complex *) input->get_buffer();
  fftwf_complex * out = (fftwf_complex *) output->get_buffer();

  // data transformation ordering is SFPT -> TFPS
  plan = fftwf_plan_dft_1d (nfft, in, out, fftw_direction, fftw_flags);
}

//! no special action required
void spip::ForwardFFTFFTW::prepare ()
{
  spip::ForwardFFT::prepare();
}

//! perform Forward FFT using FFTW
void spip::ForwardFFTFFTW::transform ()
{
  fftwf_complex * in  = (fftwf_complex *) input->get_buffer();
  fftwf_complex * out = (fftwf_complex *) output->get_buffer();

  const uint64_t nbatch = ndat / nfft;
  const uint64_t osamp_stride = nsignal * nchan * nfft * npol;

  for (unsigned isig=0; isig<nsignal; isig++)
  {
    for (unsigned ichan=0; ichan<nchan; ichan++)
    {
      for (unsigned ipol=0; ipol<npol; ipol++)
      {
        uint64_t out_samp_offset = ichan * nfft * nsignal + ipol * nsignal + isig;
        for (uint64_t ibatch=0; ibatch<nbatch; ibatch++)
        {
          fftwf_execute_dft (plan, in, out + out_samp_offset);
          in += nfft;
          out_samp_offset += osamp_stride;
        }
      }
    }
  }
}

