/***************************************************************************
 *
 *   Copyright (C) 2016 Andrew Jameson
 *   Licensed under the Academic Free License version 2.1
 *
 ***************************************************************************/

#include "spip/BatchedBackwardFFTFFTW.h"

#include <stdexcept>
#include <cmath>

using namespace std;

spip::BatchedBackwardFFTFFTW::BatchedBackwardFFTFFTW ()
{
  plan = 0;
}

spip::BatchedBackwardFFTFFTW::~BatchedBackwardFFTFFTW ()
{
  if (plan)
    fftwf_destroy_plan (plan);
  plan = 0;
}

void spip::BatchedBackwardFFTFFTW::configure ()
{
  spip::BatchedBackwardFFT::configure ();
}


void spip::BatchedBackwardFFTFFTW::prepare ()
{
  spip::BatchedBackwardFFT::prepare ();

  int fftw_direction = FFTW_BACKWARD;
  unsigned fftw_flags = FFTW_ESTIMATE;

  fftwf_complex * in  = (fftwf_complex *) input->get_buffer();
  fftwf_complex * out = (fftwf_complex *) output->get_buffer();

  // each call to the plan will perform all the backward FFTs
  // for a sample, polarisation and antenna

  //int nchan_out = nchan / nfft;
  int ndat_out  = ndat * nfft;

  int rank = 1;
  int n[1] = { nfft };
  int howmany = nchan / nfft;

  int inembed[1] = { nfft };
  int istride = npol * nsignal;   // input stride between FFT samples
  int idist = nfft * istride;     // input stride between FFT blocks

  int onembed[1] = { nfft };
  int ostride = 1;                // output stride between samples
  int odist = ndat_out * npol;    // output stride between FFT blocks

  plan = fftwf_plan_many_dft (rank, n, howmany, 
                              in, inembed, istride, idist,
                              out, onembed, ostride, odist, 
                              fftw_direction, fftw_flags);
}


//! perform BtatchedBackward FFT using FFTWF
void spip::BatchedBackwardFFTFFTW::transform ()
{
  // input and output pointers
  fftwf_complex * in  = (fftwf_complex *) input->get_buffer();
  fftwf_complex * out = (fftwf_complex *) output->get_buffer();

  // transform ordering from TFPS -> SFPT
  const uint64_t isamp_stride = nchan * npol * nsignal;

  for (uint64_t idat=0; idat<ndat; idat++)
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

    in += isamp_stride;
    out += nfft;
  }
}

