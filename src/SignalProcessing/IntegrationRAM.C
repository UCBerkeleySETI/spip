/***************************************************************************
 *
 *   Copyright (C) 2018 Andrew Jameson
 *   Licensed under the Academic Free License version 2.1
 *
 ***************************************************************************/

#include "spip/IntegrationRAM.h"

#include <iostream>
#include <cstring>
#include <stdexcept>
#include <cmath>

using namespace std;

spip::IntegrationRAM::IntegrationRAM ()
{
}

spip::IntegrationRAM::~IntegrationRAM ()
{
}

// configure the pipeline prior to runtime
void spip::IntegrationRAM::configure (spip::Ordering output_order)
{
  if (!buffer)
    buffer = new spip::ContainerRAM();
  spip::Integration::configure (output_order);

  // ensure the buffer is zero
  buffer->zero();
}


void spip::IntegrationRAM::prepare ()
{
  spip::Integration::prepare ();
}

/*
void spip::IntegrationRAM::transform_SFPT_to_SFPT ()
{
  if (verbose)
    cerr << "spip::IntegrationRAM::transform_SFPT_to_SFPT" << endl;

  float * in  = (float *) input->get_buffer();
  float * out = (float *) output->get_buffer();

  output->zero();

  const uint64_t out_pol_stride = ndat / dat_dec;
  const uint64_t out_chan_stride = (npol / pol_dec) * out_pol_stride;
  const uint64_t out_sig_stride = (nchan / chan_dec) * out_chan_stride;

  uint64_t idx = 0;
  for (unsigned isig=0; isig<nsignal; isig++)
  {
    const uint64_t out_sig_offset = (isig / signal_dec) * out_sig_stride;
    for (unsigned ichan=0; ichan<nchan; ichan++)
    {
      const uint64_t out_chan_offset = (ichan / chan_dec) * out_chan_stride;
      for (unsigned ipol=0; ipol<npol; ipol++)
      {
        const uint64_t out_pol_offset = (ipol / pol_dec) * out_pol_stride;

        uint64_t odx = out_sig_offset + out_chan_offset + out_pol_offset;
        for (uint64_t idat=0; idat<ndat; idat++)
        {
          const uint64_t odat = idat / dat_dec;
          out[odx+odat] += in[idx];
          idx++;
        }
      }
    }
  }
}
*/


void spip::IntegrationRAM::transform_TSPF_to_TSPF ()
{
  if (verbose)
    cerr << "spip::IntegrationRAM::transform_TSPF_to_TSPF" << endl;
  
  float * in  = (float *) input->get_buffer();
  float * buf = (float *) buffer->get_buffer();
  float * out = (float *) output->get_buffer();
  
  const uint64_t out_pol_stride = nchan / chan_dec;
  const uint64_t out_sig_stride = (npol / pol_dec) * out_pol_stride;
  const uint64_t out_dat_stride = (nsignal / signal_dec) * out_sig_stride;

  uint64_t idx = 0;

  for (uint64_t idat=0; idat<ndat; idat++)
  {
    const uint64_t out_dat_offset = (idat / dat_dec) * out_dat_stride;
    for (unsigned isig=0; isig<nsignal; isig++)
    { 
      const uint64_t out_sig_offset = (isig / signal_dec) * out_sig_stride;
      for (unsigned ipol=0; ipol<npol; ipol++)
      { 
        const uint64_t out_pol_offset = (ipol / pol_dec) * out_pol_stride;
        const uint64_t odx = out_dat_offset + out_sig_offset + out_pol_offset;
        for (unsigned ichan=0; ichan<nchan; ichan++)
        {
          const uint64_t ochan = ichan / chan_dec;
          buf[odx+ochan] += in[idx];
          idx++;
        }
      }
    }

    buffer_idat++;

    // we have completed a time-sample sub-integration
    if (buffer_idat == dat_dec)
    {
      // TODO remove memcpy
      memcpy (out, buf, out_dat_stride * sizeof(float));
      buffer->zero();
      buffer_idat = 0;
      out += out_dat_stride;
    }
  }
}


void spip::IntegrationRAM::transform_TFPS_to_TFPS ()
{
  if (verbose)
    cerr << "spip::IntegrationRAM::transform_TFPS_to_TFPS" << endl;

  float * in  = (float *) input->get_buffer();
  float * buf = (float *) buffer->get_buffer();
  float * out = (float *) output->get_buffer();

  const uint64_t out_pol_stride = nsignal / signal_dec;
  const uint64_t out_chan_stride = (npol / pol_dec) * out_pol_stride;
  const uint64_t out_dat_stride = (nchan / chan_dec) * out_chan_stride;

  uint64_t idx = 0;
  for (uint64_t idat=0; idat<ndat; idat++)
  {
    const uint64_t out_dat_offset = (idat / dat_dec) * out_dat_stride;
    for (unsigned ichan=0; ichan<nchan; ichan++)
    {
      const uint64_t out_chan_offset = (ichan / chan_dec) * out_chan_stride;
      for (unsigned ipol=0; ipol<npol; ipol++)
      {
        const uint64_t out_pol_offset = (ipol / pol_dec) * out_pol_stride;
        const uint64_t odx = out_dat_offset + out_chan_offset + out_pol_offset;

        for (unsigned isig=0; isig<nsignal; isig++)
        {
          const unsigned osig = isig / nsignal;
          buf[odx+osig] += in[idx];
          idx++;
        }
      }
    }

    buffer_idat++;

    // we have completed a time-sample sub-integration
    if (buffer_idat == dat_dec)
    {
      // TODO remove memcpy
      memcpy (out, buf, out_dat_stride * sizeof (float));
      buffer->zero();
      buffer_idat = 0;
      out += out_dat_stride;
    }
  }
}
