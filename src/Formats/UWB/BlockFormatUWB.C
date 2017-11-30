/***************************************************************************
 *
 *   Copyright (C) 2015 Andrew Jameson
 *   Licensed under the Academic Free License version 2.1
 *
 ***************************************************************************/

#include "spip/BlockFormatUWB.h"
#include <math.h>

#include <cstdlib>
#include <cstdio>
#include <iostream>

using namespace std;

spip::BlockFormatUWB::BlockFormatUWB()
{
  nchan = 1;
  npol = 2;
  ndim = 2;
  nbit = 16;

#ifdef HAVE_FFTW3
  in = NULL;
  out = NULL;
  plan = NULL;
#endif
}

spip::BlockFormatUWB::~BlockFormatUWB()
{
#ifdef HAVE_FFTW3
  deconfigure_fft();
#endif
}

#ifdef HAVE_FFTW3
void spip::BlockFormatUWB::deconfigure_fft ()
{
  if (in)
    fftwf_free (in);
  in = 0;
  if (out)
    fftwf_free (out);
  out = 0;
  if (plan)
    fftwf_destroy_plan (plan);
  plan = 0;
}

void spip::BlockFormatUWB::configure_fft ()
{
  // ensure FFT resources are not allocated
  deconfigure_fft ();

  // this depends on how many output channels are requested
  nfft = nfreq_ft / nchan;

  // configure simple fft
  in = (fftwf_complex *) fftwf_malloc (sizeof(fftwf_complex) * nfft);
  out = (fftwf_complex *) fftwf_malloc (sizeof(fftwf_complex) * nfft);

  int fftw_direction = FFTW_FORWARD;
  int fftw_flags = FFTW_ESTIMATE;
  plan = fftwf_plan_dft_1d (nfft, in, out, fftw_direction, fftw_flags);
}
#endif

void spip::BlockFormatUWB::unpack_hgft (char * buffer, uint64_t nbytes)
{
  const unsigned nsamp = nbytes / bytes_per_sample;
  const unsigned nsamp_per_time = nsamp / ntime;

  int16_t * input = (int16_t *) buffer;

  // always zero since UWB has 1 channel only
  unsigned ifreq_hg = 0;
  unsigned ival = 0;
  unsigned ibin, itime, power, ifreq;
  int re, im;
  float ref, imf;

  for (unsigned isamp=0; isamp<nsamp; isamp++)
  {
    for (unsigned ipol=0; ipol<npol; ipol++)
    {
      re = (int) input[isamp];
      im = (int) input[isamp+1];

      ref = (float) re;
      imf = (float) im;

      sums[ipol*ndim + 0] += ref;
      sums[ipol*ndim + 1] += imf;

      // histograms only have 256 bins
      ibin = (re + 32768) / 256;
      hist[ipol][0][ifreq_hg][ibin]++;

      ibin = (im + 32768) / 256;
      hist[ipol][1][ifreq_hg][ibin]++;

#ifdef HAVE_FFTW3
      in[ival][0] = ref;
      in[ival][1] = imf;
      ival++;

      if (ival == nfft)
      {
        itime = isamp / nsamp_per_time;
        fftwf_execute_dft (plan, in, out);
        for (ifreq=0; ifreq<nfreq_ft; ifreq++)
        {
          ref = out[ifreq][0];
          imf = out[ifreq][1];
          power = (unsigned) ((ref * ref) + (imf * imf));
          freq_time[ipol][ifreq][itime] += power;
        }
        ival = 0;
      }
#else
      power = (unsigned) ((ref * ref) + (imf * imf));
      freq_time[ipol][ifreq_hg][itime] += power;
#endif
      isamp += 2;
    }
  }
}



void spip::BlockFormatUWB::unpack_ms(char * buffer, uint64_t nbytes)
{
  const unsigned nsamp = nbytes / bytes_per_sample;
  float ndat = (float) (nsamp * nchan);

  for (unsigned i=0; i<npol * ndim; i++)
    means[i] = sums[i] / ndat;

  int16_t * in = (int16_t *) buffer;
  uint64_t isamp = 0;
  int re, im;
  float diff;

#ifdef _DEBUG
  cerr << "spip::BlockFormatUWB::unpack_ms nsamp=" << nsamp << endl;
  cerr << "spip::BlockFormatUWB::unpack_ms nchan_per_freq=" << nchan_per_freq << " nblock=" << nblock << end
l;
#endif

  for (unsigned isamp=0; isamp<nsamp; isamp++)
  {
    for (unsigned ipol=0; ipol<npol; ipol++)
    {
      unsigned idx = ipol*ndim + 0;
      re = (int) in[isamp];
      diff = (float) re - means[idx];
      variances[idx] += diff * diff;

      idx = ipol*ndim + 1;
      im = (int) in[isamp+1];
      diff = (float) im - means[idx];
      variances[idx] += diff * diff;

      isamp += 2;
    }
  }

  for (unsigned i=0; i<npol * ndim; i++)
  {
    variances[i] /= ndat;
    stddevs[i] = sqrtf (variances[i]);
  }
}
