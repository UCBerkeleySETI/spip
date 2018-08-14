/***************************************************************************
 *
 *   Copyright (C) 2015 Andrew Jameson
 *   Licensed under the Academic Free License version 2.1
 *
 ***************************************************************************/

#include "spip/BlockFormatMeerKAT.h"
#include <math.h>

#include <cstdlib>
#include <cstdio>
#include <iostream>

using namespace std;

spip::BlockFormatMeerKAT::BlockFormatMeerKAT(int _nchan)
{
  nbin = 256;
  nchan = _nchan;
  npol = 2;
  ndim = 2;
  nbit = 8;
}

spip::BlockFormatMeerKAT::~BlockFormatMeerKAT()
{
}

void spip::BlockFormatMeerKAT::unpack_hgft (char * buffer, uint64_t nbytes)
{
  const unsigned nsamp = nbytes / bytes_per_sample;
  const unsigned nsamp_per_time = nsamp / ntime;
  const unsigned nchan_per_freq_hg = nchan / nfreq_hg;
  const unsigned nchan_per_freq_ft = nchan / nfreq_ft;
  const unsigned nsamp_block = resolution / (nchan * npol * ndim * nbit / 8);
  const unsigned nblock = nsamp / nsamp_block;

  int8_t * in = (int8_t *) buffer;
  uint64_t idat = 0;
  unsigned ibin, ifreq_hg, ifreq_ft, itime;
  int re, im;
  unsigned power;

#ifdef _DEBUG
  cerr << "spip::BlockFormatMeerKAT::unpack_hgft nchan=" << nchan << " nfreq_ft=" << nfreq_ft << endl;

  cerr << "spip::BlockFormatMeerKAT::unpack_hgft nsamp=" << nsamp << " nsamp_per_time=" << nsamp_per_time << " nchan_per_freq_hg=" << nchan_per_freq_hg << " nchan_per_freq_ft=" << nchan_per_freq_ft << " nsamp_block=" << nsamp_block << " nblock=" << nblock << endl;
#endif

  for (unsigned iblock=0; iblock<nblock; iblock++)
  {
    for (unsigned ipol=0; ipol<npol; ipol++)
    {
      for (unsigned ichan=0; ichan<nchan; ichan++)
      {
        ifreq_hg = ichan / nchan_per_freq_hg;
        ifreq_ft = ichan / nchan_per_freq_ft;

        for (unsigned isamp=0; isamp<nsamp_block; isamp++)
        {
#ifdef _DEBUG
          cerr << "[" << iblock << "][" << ipol << "][" << ichan << "][" << isamp << "]" << endl;
#endif
          re = (int) in[idat];
          im = (int) in[idat+1];

          sums[ipol*ndim + 0] += (float) re;
          sums[ipol*ndim + 1] += (float) im;

          ibin = re + 128;
          hist[ipol][0][ifreq_hg][ibin]++;

          ibin = im + 128;
          hist[ipol][1][ifreq_hg][ibin]++;

          // detect and average the time samples into a NPOL sets of NCHAN * 512 waterfalls
          power = (unsigned) ((re * re) + (im * im));
          itime = ((iblock * nsamp_block) + isamp) / nsamp_per_time;
          freq_time[ipol][ifreq_ft][itime] += power;

          idat += 2;
        }
      }
    }
  }
}



void spip::BlockFormatMeerKAT::unpack_ms(char * buffer, uint64_t nbytes)
{
  const unsigned nsamp = nbytes / bytes_per_sample;
  const unsigned nsamp_block = resolution / (nchan * npol * ndim * nbit / 8);
  const unsigned nblock = nsamp / nsamp_block;

  float ndat = (float) (nsamp * nchan);

  for (unsigned i=0; i<npol * ndim; i++)
    means[i] = sums[i] / ndat;

  int8_t * in = (int8_t *) buffer;
  uint64_t idat = 0;
  int re, im;
  float diff;

  for (unsigned iblock=0; iblock<nblock; iblock++)
  {
    for (unsigned ipol=0; ipol<npol; ipol++)
    {
      for (unsigned ichan=0; ichan<nchan; ichan++)
      {
        for (unsigned isamp=0; isamp<nsamp_block; isamp++)
        {
          unsigned idx = ipol*ndim + 0;
          re = (int) in[idat];
          diff = (float) re - means[idx];
          variances[idx] += diff * diff;

          idx = ipol*ndim + 1;
          im = (int) in[idat+1];
          diff = (float) im - means[idx];
          variances[idx] += diff * diff;

          idat += 2;
        }
      }
    }
  }

  for (unsigned i=0; i<npol * ndim; i++)
  {
    variances[i] /= ndat;
    stddevs[i] = sqrtf (variances[i]);
  }
}
