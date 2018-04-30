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

//#define _DEBUG

using namespace std;

spip::BlockFormatUWB::BlockFormatUWB()
{
  nchan = 1;
  npol = 2;
  ndim = 2;
  nbit = 16;
  nbin = 65536;

  // Ask AJ...
  //scale = 0.022325;
  scale = 1;

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
#ifdef _DEBUG
  cerr << "spip::BlockFormatUWB::configure_fft() nfreq_ft=" << nfreq_ft << endl;
#endif
  // ensure FFT resources are not allocated
  deconfigure_fft ();

  // this depends on how many output channels are requested
  nfft = nfreq_ft / nchan;

  // increment the sampling time
  tsamp *= nfft;

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
  if (!plan)
    configure_fft();

#ifdef _DEBUG
  cerr << "spip::BlockFormatUWB::unpack_hgft nbytes=" << nbytes 
       << " bytes_per_sample=" << bytes_per_sample << endl;
#endif
  const uint64_t nsamp = nbytes / bytes_per_sample;
  const uint64_t nsamp_per_block = 2048;
  const uint64_t nblock = nsamp / nsamp_per_block;

#ifdef _DEBUG
  cerr << "spip::BlockFormatUWB::unpack_hgft nsamp=" << nsamp 
       << " nsamp_per_block=" << nsamp_per_block << " nblock=" << nblock 
       << " ntime=" << ntime << endl;
#endif
  const unsigned nsamp_per_time = nsamp / ntime;
#ifdef _DEBUG
  cerr << "spip::BlockFormatUWB::unpack_hgft nsamp=" << nsamp << " ntime=" << ntime << " nsamp_per_time=" << nsamp_per_time << endl;
#endif

  int16_t * input = (int16_t *) buffer;

  // always zero since UWB has 1 channel only
  uint64_t idx = 0;
  unsigned ifreq_hg = 0;
  unsigned ival = 0;
  unsigned ibin, itime, ifreq;
  int16_t re, im;
  float ref, imf, power;

  const uint64_t pol_stride = ndim * nsamp_per_block;
  const uint64_t block_stride = npol * pol_stride;;

  // data are oranise in blocks of 2048 samples from pol0, 2048 pol1, etc
  for (unsigned iblock=0; iblock<nblock; iblock++)
  {
    for (unsigned ipol=0; ipol<npol; ipol++)
    {
      idx = (iblock * block_stride) + (ipol * pol_stride);
      for (unsigned ibit=0; ibit<nsamp_per_block; ibit++)
      {
        re = convert_offset_binary(input[idx+0]);
        im = convert_offset_binary(input[idx+1]);

        ref = float(re);
        imf = float(im);

        sums[ipol*ndim + 0] += ref;
        sums[ipol*ndim + 1] += imf;

        // histograms have 65536 bins
        ibin = (int32_t(re) + 32768);
        hist[ipol][0][ifreq_hg][ibin]++;

        ibin = (int32_t(im) + 32768);
        hist[ipol][1][ifreq_hg][ibin]++;

#ifdef HAVE_FFTW3
        // build a FFT input block of length nfft
        in[ival][0] = ref;
        in[ival][1] = imf;
        ival++;
  
        if (ival == nfft)
        {
          // the absolute sample number for the start of the block
          unsigned isamp = iblock * nsamp_per_block + ibit;

          // the output time bin number for the start of the blocks
          itime = isamp / nsamp_per_time;

          // perform the FFT to generate nfft output channels in out
          fftwf_execute_dft (plan, in, out);

          // re-arrange the channels
          for (ifreq=0; ifreq<nfreq_ft; ifreq++)
          {
            unsigned ofreq = (ifreq + (nfreq_ft/2)) % nfreq_ft;
            ref = out[ifreq][0] / nfft;
            imf = out[ifreq][1] / nfft;
            power = ((ref * ref) + (imf * imf));
            freq_time[ipol][ofreq][itime] += power;
          }
          ival = 0;
        }
#else
        power = ((ref * ref) + (imf * imf));
        freq_time[ipol][ifreq_hg][itime] += power;
#endif
        idx += 2;
      } 
    }
  }
}

void spip::BlockFormatUWB::unpack_ms(char * buffer, uint64_t nbytes)
{
  const uint64_t nsamp = nbytes / bytes_per_sample;
  const uint64_t nsamp_per_block = 2048;
  const uint64_t max_nblock = 4;
  const uint64_t nblock = std::min(nsamp / nsamp_per_block, max_nblock);
  float ndat = (float) (nsamp * nchan);

  for (unsigned i=0; i<npol * ndim; i++)
    means[i] = sums[i] / ndat;

  int16_t * in = (int16_t *) buffer;
  int re, im;
  float diff;
  uint64_t idx;

#ifdef _DEBUG
  cerr << "spip::BlockFormatUWB::unpack_ms nsamp=" << nsamp << " nblock=" << nblock << " ndat=" << ndat << endl;
#endif

  uint64_t isamp = 0;

  for (unsigned iblock=0; iblock<nblock; iblock++)
  {
    for (unsigned ipol=0; ipol<npol; ipol++)
    {
      for (unsigned ibit=0; ibit<nsamp_per_block; ibit++)
      {
        re = int(convert_offset_binary(in[isamp+0]));
        im = int(convert_offset_binary(in[isamp+1]));

        // real
        idx = ipol*ndim + 0;
        diff = (float) re - means[idx];
        variances[idx] += diff * diff;

        // imag
        idx = ipol*ndim + 1;
        diff = (float) im - means[idx];
        variances[idx] += diff * diff;

        isamp += 2;
      }
    }
  }

  for (unsigned i=0; i<npol * ndim; i++)
  {
    variances[i] /= ndat;
    stddevs[i] = sqrtf (variances[i]);
#ifdef _DEBUG
    cerr << "spip::BlockFormatUWB::unpack_ms ipoldim=" << i << " mean=" << means[i] << " variance=" << variances[i] << " stddev=" << stddevs[i] << endl;
#endif
  }
}
