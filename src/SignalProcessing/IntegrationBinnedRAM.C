/***************************************************************************
 *
 *   Copyright (C) 2018 Andrew Jameson
 *   Licensed under the Academic Free License version 2.1
 *
 ***************************************************************************/

#include "spip/IntegrationBinnedRAM.h"

#include <iostream>
#include <cstring>
#include <stdexcept>
#include <cmath>

using namespace std;

spip::IntegrationBinnedRAM::IntegrationBinnedRAM ()
{
}

spip::IntegrationBinnedRAM::~IntegrationBinnedRAM ()
{
  if (buffer)
    delete buffer;
  buffer = NULL;

  if (binplan)
    delete binplan;
  binplan = NULL;
}

// configure the pipeline prior to runtime
void spip::IntegrationBinnedRAM::configure (spip::Ordering output_order)
{
  if (!buffer)
    buffer = new spip::ContainerRAM();

  if (!binplan)
    binplan = new spip::ContainerRAM();

  spip::IntegrationBinned::configure (output_order);

  // ensure the buffer is zero
  buffer->zero();
}

// configure the pipeline prior to runtime
void spip::IntegrationBinnedRAM::prepare_binplan ()
{
  // resize the bin plan to be large enough
  if (verbose)
    cerr << "spip::IntegrationBinnedRAM::prepare_binplan binplan.resize (" << ndat << ")" << endl;
  binplan->set_ndat(ndat);
  binplan->resize ();

  uint64_t end_idat = start_idat + ndat;

  if (verbose)
    cerr << "spip::IntegrationBinnedRAM::prepare_binplan start_idat=" << start_idat << " end_idat=" << end_idat <<
endl;
  int bin;
  int * bp = (int *) binplan->get_buffer();

#ifdef _DEBUG
  cerr << "spip::IntegrationBinnedRAM::prepare_binplan idat start=" << start_idat 
       << " end=" << end_idat << " ndat=" << ndat << endl;
  cerr << "spip::IntegrationBinnedRAM::prepare_binplan tsamp=" << tsamp 
       << " cal_epoch_delta=" << cal_epoch_delta << " cal_period=" 
       << cal_period << " cal_phase=" << cal_phase << endl;
#endif

  for (uint64_t idat=start_idat; idat < end_idat; idat++)
  {
    // convert the idat to an offset time from the utc_start [seconds]
    double idat_offset_utc_start_beg = (tsamp * idat) / 1000000;
    double idat_offset_utc_start_end = (tsamp * (idat+1)) / 1000000;

    // convert to offset from cal epoch [seconds]
    double idat_offset_cal_epoch_beg = idat_offset_utc_start_beg + (cal_epoch_delta);
    double idat_offset_cal_epoch_end = idat_offset_utc_start_end + (cal_epoch_delta);

    // convert to phase of the cal
    double phi_beg = (fmod (idat_offset_cal_epoch_beg, cal_period) / cal_period) - cal_phase;
    double phi_end = (fmod (idat_offset_cal_epoch_end, cal_period) / cal_period) - cal_phase;

    // bin 0 == OFF bin 1 == ON
    // if the starting phase is greater than the duty cycle
    if ((phi_beg > 0) && (phi_end < cal_duty_cycle))
      bin = 1;
    else if ((phi_beg > cal_duty_cycle) && (phi_end < 1))
      bin = 0;
    else if ((phi_beg < 0 ) && (phi_end < 0))
      bin = 0;
    else
      bin = -1;

    bp[idat-start_idat] = bin;

#ifdef _DEBUG
    if ((idat == start_idat) || (idat == (end_idat-1)))
      cerr << "spip::IntegrationBinnedRAM::prepare_binplan idat=" << idat << " ibin=" << bin << " idat_offset=" << idat_offset_utc_start_beg << endl;
#endif

  }

  // for the next iteration
  start_idat = end_idat;
}


void spip::IntegrationBinnedRAM::transform_TSPF_to_TSPFB ()
{
  if (verbose)
    cerr << "spip::IntegrationBinnedRAM::transform_TSPF_to_TSPFB" << endl;
  
  float * in  = (float *) input->get_buffer();
  float * buf = (float *) buffer->get_buffer();
  int   * bp = (int *) binplan->get_buffer();
  float * out = (float *) output->get_buffer();
  
  const uint64_t out_chan_stride = nbin;
  const uint64_t out_pol_stride  = (nchan / chan_dec) * out_chan_stride;
  const uint64_t out_sig_stride  = (npol / pol_dec) * out_pol_stride;
  const uint64_t out_dat_stride  = (nsignal / signal_dec) * out_sig_stride;

  if (verbose)
    cerr << "spip::IntegrationBinnedRAM::transform_TSPF_to_TSPFB ndat=" << ndat << " nbin=" << nbin << " npol=" << npol << " nsig=" << nsignal << endl;

  uint64_t idx = 0;

  for (uint64_t idat=0; idat<ndat; idat++)
  {
    const int ibin = bp[idat];
    const uint64_t out_dat_offset = ((idat / dat_dec) * out_dat_stride) + ibin;
    for (unsigned isig=0; isig<nsignal; isig++)
    {
      const uint64_t out_sig_offset = out_dat_offset + ((isig / signal_dec) * out_sig_stride);
      for (unsigned ipol=0; ipol<npol; ipol++)
      { 
        const uint64_t out_pol_offset = out_sig_offset + ((ipol / pol_dec) * out_pol_stride);
        for (unsigned ichan=0; ichan<nchan; ichan++)
        {
          // negative ibin are not to be binned;
          if (ibin >= 0)
          {
            const uint64_t odx = out_pol_offset + ((ichan / chan_dec) * out_chan_stride);
            buf[odx] += in[idx];
          }
          idx++;
        }
      }
    }

    buffer_idat++;

    // we have completed a time-sample sub-integration
    if (buffer_idat == dat_dec)
    {
      if (verbose)
        cerr << "spip::IntegrationBinnedRAM::transform_TSPF_to_TSPFB memcpy(" 
             << (void *) out << "," << (void *) buf << "," 
             << out_dat_stride * sizeof(float) << ")" << endl;

      // TODO remove memcpy
      memcpy (out, buf, out_dat_stride * sizeof(float));
      buffer->zero();
      buffer_idat = 0;
      out += out_dat_stride;
    }
  }
}
