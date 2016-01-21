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

spip::BlockFormatMeerKAT::BlockFormatMeerKAT()
{
  cerr << "spip::BlockFormatMeerKAT::BlockFormatMeerKAT()" << endl;

  nchan = 1024;
  npol = 2;
  ndim = 2;
  nbit = 8;

  nsamp_block = 128;
}

spip::BlockFormatMeerKAT::~BlockFormatMeerKAT()
{
  cerr << "spip::BlockFormatMeerKAT::~BlockFormatMeerKAT()" << endl;
}

void spip::BlockFormatMeerKAT::unpack_hgft (char * buffer, uint64_t nbytes)
{
  const unsigned nsamp = nbytes / bytes_per_sample;
  const unsigned nsamp_per_time = nsamp / ntime;
  const unsigned nchan_per_freq = nchan / nfreq;
  const unsigned nblock = nsamp / nsamp_block;

  int8_t * in = (int8_t *) buffer;
  int8_t val;

  uint64_t idat = 0;
  unsigned ibin, ifreq, itime;
  int re, im;
  unsigned power;

  for (unsigned iblock=0; iblock<nblock; iblock++)
  {
    for (unsigned ipol=0; ipol<npol; ipol++)
    {
      for (unsigned ichan=0; ichan<nchan; ichan++)
      {
        ifreq = ichan / nchan_per_freq;

        for (unsigned isamp=0; isamp<nsamp_block; isamp++)
        {
          re = (int) in[idat];
          im = (int) in[idat+1];

          sums[ipol*ndim + 0] += (float) re;
          sums[ipol*ndim + 1] += (float) im;

          ibin = re + 128;
          hist[ipol][0][ibin]++;

          ibin = im + 128;
          hist[ipol][1][ibin]++;

          // detect and average the timesamples into a NPOL sets of NCHAN * 512 waterfalls
          power = (unsigned) ((re * re) + (im * im));
          itime = isamp / nsamp_per_time;
          freq_time[ipol][ifreq][itime] += power;

          idat += 2;
        }
      }
    }
  }
}



void spip::BlockFormatMeerKAT::unpack_ms(char * buffer, uint64_t nbytes)
{
  const unsigned nsamp = nbytes / bytes_per_sample;
  const unsigned nsamp_per_time = nsamp / ntime;
  const unsigned nchan_per_freq = nchan / nfreq;
  const unsigned nblock = nsamp / nsamp_block;

  float ndat = (float) (nsamp * nchan);

  for (unsigned i=0; i<npol * ndim; i++)
    means[i] = sums[i] / ndat;

  int8_t * in = (int8_t *) buffer;
  int8_t val;

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