/***************************************************************************
 *
 *   Copyright (C) 2015 Andrew Jameson
 *   Licensed under the Academic Free License version 2.1
 *
 ***************************************************************************/

#include "spip/BlockFormat.h"
#include "spip/Error.h"

#include <iostream>
#include <fstream>
#include <cstring>
#include <cstdio>
#include <unistd.h>
#include <math.h>

#ifdef __cplusplus
#define __STDC_CONSTANT_MACROS
#ifdef _STDINT_H
#undef _STDINT_H
#endif
#include <stdint.h>
#endif

using namespace std;

spip::BlockFormat::BlockFormat()
{
  npol = 1;
  ndim = 1;
  nchan = 1;
  nbit = 8;
  freq = 0;
  bw = 0;

  temp_file = (char *) malloc(FILENAME_MAX);
}

spip::BlockFormat::~BlockFormat()
{
  if (temp_file)
    free (temp_file);
  temp_file = NULL;
}

void spip::BlockFormat::prepare (unsigned _nbin, unsigned _ntime,
                                 unsigned _nfreq, double _freq,
                                 double _bw, double _tsamp)
{
  bits_per_sample = nchan * npol * ndim * nbit;
  bytes_per_sample = bits_per_sample / 8;

  nbin = _nbin;
  ntime = _ntime;
  bw = _bw;
  freq = _freq;
  tsamp = _tsamp;

  // configure the number of channels in the FT
  if (_nfreq < nchan)
  {
    nfreq_ft = nchan;
    while (_nfreq <= nfreq_ft)
    {
      nfreq_ft /= 2;
    }
  }
  else
  {
    nfreq_ft = nchan;
    while (_nfreq > nfreq_ft)
      nfreq_ft *= 2;
  }

  // configure the number of channels in the HG 
  nfreq_hg = nchan;
  while (_nfreq <= nfreq_hg)
  {
    nfreq_hg /= 2;
  }

  sums.resize (npol * ndim);
  means.resize (npol * ndim);
  variances.resize (npol * ndim);
  stddevs.resize (npol * ndim);

  freq_time.resize(npol);
  bandpass.resize(npol);
  ts_min.resize(npol);
  ts_mean.resize(npol);
  ts_rms.resize(npol);
  ts_max.resize(npol);
  ts_sum.resize(npol);
  ts_sumsq.resize(npol);
  hist.resize(npol);

  for (unsigned ipol=0; ipol<npol; ipol++)
  {
    freq_time[ipol].resize(nfreq_ft);
    bandpass[ipol].resize(nfreq_ft);
    hist[ipol].resize(ndim);
    ts_min[ipol].resize(ntime);
    ts_mean[ipol].resize(ntime);
    ts_rms[ipol].resize(ntime);
    ts_max[ipol].resize(ntime);
    ts_sum[ipol].resize(ntime);
    ts_sumsq[ipol].resize(ntime);
    for (unsigned idim=0; idim<ndim; idim++)
    {
      hist[ipol][idim].resize(nfreq_hg);
    }
    for (unsigned ifreq=0; ifreq<nfreq_ft; ifreq++)
    {
      freq_time[ipol][ifreq].resize(ntime);
    }
    for (unsigned ifreq=0; ifreq<nfreq_hg; ifreq++)
    {
      for (unsigned idim=0; idim<ndim; idim++)
      {
        hist[ipol][idim][ifreq].resize(nbin);
      }
    }
  }

}

void spip::BlockFormat::reset()
{
  for (unsigned ipol=0; ipol<npol; ipol++)
  {
    fill (ts_min[ipol].begin(), ts_min[ipol].end(), 1e9);
    fill (ts_mean[ipol].begin(), ts_mean[ipol].end(), 0);
    fill (ts_max[ipol].begin(), ts_max[ipol].end(), -1e9);
    fill (ts_sum[ipol].begin(), ts_sum[ipol].end(), 0);
    fill (ts_sumsq[ipol].begin(), ts_sumsq[ipol].end(), 0);
    fill (bandpass[ipol].begin(), bandpass[ipol].end(), 0);
    for (unsigned ifreq=0; ifreq<nfreq_ft; ifreq++)
    {
      fill(freq_time[ipol][ifreq].begin(), freq_time[ipol][ifreq].end(), 0);
    }
    for (unsigned ifreq=0; ifreq<nfreq_hg; ifreq++)
    {
      for (unsigned idim=0; idim<ndim; idim++)
      {
        fill ( hist[ipol][idim][ifreq].begin(), hist[ipol][idim][ifreq].end(), 0);
      }
    }
  }

  // zero sums and variances
  fill (sums.begin(), sums.end(), 0);
  fill (variances.begin(), variances.end(), 0);
}

void spip::BlockFormat::write_histograms(string hg_filename)
{
  // create a temporary filename tempplate  in the specified directory
  sprintf (temp_file, "%sXXXXXX", hg_filename.c_str());
  int fd = mkstemp (temp_file);
  close (fd);

  ofstream hg_file (temp_file, ofstream::binary);

  hg_file.write (reinterpret_cast<const char *>(&npol), sizeof(npol));
  hg_file.write (reinterpret_cast<const char *>(&nfreq_hg), sizeof(nfreq_hg));
  hg_file.write (reinterpret_cast<const char *>(&ndim), sizeof(ndim));
  hg_file.write (reinterpret_cast<const char *>(&nbin), sizeof(nbin));
  for (unsigned ipol=0; ipol<npol; ipol++)
  {
    for (unsigned idim=0; idim<ndim; idim++)
    {
      for (unsigned ifreq=0; ifreq<nfreq_hg; ifreq++)
      {  
        const char * buffer = reinterpret_cast<const char *>(&hist[ipol][idim][ifreq][0]);
        hg_file.write(buffer, hist[ipol][idim][ifreq].size() * sizeof(unsigned));
      }
    }
  }
  hg_file.close();

  if (rename(temp_file, hg_filename.c_str()) != 0)
  {
    throw Error (InvalidState, "spip::BlockFormat::write_histograms",
                 "failed to reame temporary file to actual");
  }
}

void spip::BlockFormat::write_freq_times(string ft_filename)
{
  // create a temporary filename tempplate  in the specified directory
  sprintf (temp_file, "%sXXXXXX", ft_filename.c_str());
  int fd = mkstemp (temp_file);
  close (fd);

  ofstream ft_file (temp_file, ofstream::binary);
  ft_file.write (reinterpret_cast<const char *>(&npol), sizeof(npol));
  ft_file.write (reinterpret_cast<const char *>(&nfreq_ft), sizeof(nfreq_ft));
  ft_file.write (reinterpret_cast<const char *>(&ntime), sizeof(ntime));
  ft_file.write (reinterpret_cast<const char *>(&freq), sizeof(freq));
  ft_file.write (reinterpret_cast<const char *>(&bw), sizeof(bw));
  ft_file.write (reinterpret_cast<const char *>(&tsamp), sizeof(tsamp));
  for (unsigned ipol=0; ipol<npol; ipol++)
  {
    for (unsigned ifreq=0; ifreq<nfreq_ft; ifreq++)
    {
      const char * buffer = reinterpret_cast<const char*>(&freq_time[ipol][ifreq][0]);
      ft_file.write (buffer, freq_time[ipol][ifreq].size() * sizeof(float));
    }
  }
  ft_file.close();

  if (rename(temp_file, ft_filename.c_str()) != 0)
  {
    throw Error (InvalidState, "spip::BlockFormat::write_freq_times",
                 "failed to reame temporary file to actual");
  }
}

void spip::BlockFormat::write_bandpasses (string bp_filename)
{
  // create a temporary filename tempplate  in the specified directory
  sprintf (temp_file, "%sXXXXXX", bp_filename.c_str());
  int fd = mkstemp (temp_file);
  close (fd);

  ofstream bp_file (temp_file, ofstream::binary);
  bp_file.write (reinterpret_cast<const char *>(&npol), sizeof(npol));
  bp_file.write (reinterpret_cast<const char *>(&nfreq_ft), sizeof(nfreq_ft));
  bp_file.write (reinterpret_cast<const char *>(&freq), sizeof(freq));
  bp_file.write (reinterpret_cast<const char *>(&bw), sizeof(bw));
  for (unsigned ipol=0; ipol<npol; ipol++)
  {
    const char * buffer = reinterpret_cast<const char*>(&bandpass[ipol][0]);
    bp_file.write (buffer, bandpass[ipol].size() * sizeof(float));
  }
  bp_file.close();

  if (rename(temp_file, bp_filename.c_str()) != 0)
  {
    throw Error (InvalidState, "spip::BlockFormat::write_bandpass",
                 "failed to reame temporary file to actual");
  }
}

void spip::BlockFormat::write_time_series (string ts_filename)
{
  // create a temporary filename tempplate  in the specified directory
  sprintf (temp_file, "%sXXXXXX", ts_filename.c_str());
  int fd = mkstemp (temp_file);
  close (fd);

  ofstream ts_file (temp_file, ofstream::binary);
  ts_file.write (reinterpret_cast<const char *>(&npol), sizeof(npol));
  ts_file.write (reinterpret_cast<const char *>(&ntime), sizeof(ntime));
  ts_file.write (reinterpret_cast<const char *>(&tsamp), sizeof(tsamp));
  for (unsigned ipol=0; ipol<npol; ipol++)
  {
    const char * buffer = reinterpret_cast<const char*>(&ts_min[ipol][0]);
    ts_file.write (buffer, ts_min[ipol].size() * sizeof(float));
    buffer = reinterpret_cast<const char*>(&ts_mean[ipol][0]);
    ts_file.write (buffer, ts_mean[ipol].size() * sizeof(float));
    buffer = reinterpret_cast<const char*>(&ts_max[ipol][0]);
    ts_file.write (buffer, ts_max[ipol].size() * sizeof(float));
  }
  ts_file.close();

  if (rename(temp_file, ts_filename.c_str()) != 0)
  {
    throw Error (InvalidState, "spip::BlockFormat::write_time_series",
                 "failed to reame temporary file to actual");
  }
}

void spip::BlockFormat::write_mean_stddevs(string ms_filename)
{
  // create a temporary filename tempplate  in the specified directory
  sprintf (temp_file, "%sXXXXXX", ms_filename.c_str());
  int fd = mkstemp (temp_file);
  close (fd);
  
  ofstream ms_file (temp_file, ofstream::binary);

  ms_file.write (reinterpret_cast<const char *>(&npol), sizeof(npol));
  ms_file.write (reinterpret_cast<const char *>(&ndim), sizeof(ndim));
  {
    const char * buffer;
    buffer = reinterpret_cast<const char*>(&means[0]);
    ms_file.write (buffer, means.size() * sizeof(float));
    buffer = reinterpret_cast<const char*>(&stddevs[0]);
    ms_file.write (buffer, stddevs.size() * sizeof(float));
  }
  ms_file.close();
  if (rename(temp_file, ms_filename.c_str()) != 0)
  {
    throw Error (InvalidState, "spip::BlockFormat::write_mean_stddevs",
                 "failed to reame temporary file to actual");
  }

}
