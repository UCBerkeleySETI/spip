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
}

// configure the pipeline prior to runtime
void spip::IntegrationBinnedRAM::configure (spip::Ordering output_order)
{
  if (!buffer)
    buffer = new spip::ContainerRAM();
  spip::IntegrationBinned::configure (output_order);

  // ensure the buffer is zero
  buffer->zero();
}

void spip::IntegrationBinnedRAM::transform_TSPF_to_TSPFB ()
{
  if (verbose)
    cerr << "spip::IntegrationBinnedRAM::transform_TSPF_to_TSPFB" << endl;
  
  float * in  = (float *) input->get_buffer();
  float * buf = (float *) buffer->get_buffer();
  float * out = (float *) output->get_buffer();
  
  const uint64_t out_chan_stride = nbin;
  const uint64_t out_pol_stride  = (nchan / chan_dec) * out_chan_stride;
  const uint64_t out_sig_stride  = (npol / pol_dec) * out_pol_stride;
  const uint64_t out_dat_stride  = (nsignal / signal_dec) * out_sig_stride;

  uint64_t idx = 0;

  for (uint64_t idat=0; idat<ndat; idat++)
  {
    const int ibin = binplan[idat];
    const uint64_t out_dat_offset = (idat / dat_dec) * out_dat_stride + ibin;
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
            const uint64_t odx = out_pol_offset + (ichan / chan_dec); 
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
