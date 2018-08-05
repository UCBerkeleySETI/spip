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

#ifdef _DEBUG
  cerr << "spip::BlockFormat::prepare nbin=" << nbin << " ntime=" << ntime << " nfreq=" << _nfreq << " bw=" << bw << " freq=" << freq << " tsamp=" << tsamp << " nchan=" << nchan << endl;
#endif

  // configure the number of channels in the FT
  if (_nfreq < nchan)
  {
    nfreq_ft = nchan;
    while (_nfreq < nfreq_ft)
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
  while (_nfreq < nfreq_hg)
  {
    nfreq_hg /= 2;
  }
  // frequency resolved histograms formed from channelisers are not all that useful
  nfreq_hg = 1;


  sums.resize (npol * ndim);
  means.resize (npol * ndim);
  variances.resize (npol * ndim);
  stddevs.resize (npol * ndim);

  freq_time.resize(npol);
  hist.resize(npol);

#ifdef _DEBUG
  cerr << "spip::BlockFormat::prepare hist [" << npol << "][" << ndim << "][" << nfreq_hg << "][" << nbin << "]" << endl;
#endif

  for (unsigned ipol=0; ipol<npol; ipol++)
  {
    freq_time[ipol].resize(nfreq_ft);
    hist[ipol].resize(ndim);
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
