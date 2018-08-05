/***************************************************************************
 *
 *   Copyright (C) 2018 Andrew Jameson
 *   Licensed under the Academic Free License version 2.1
 *
 ***************************************************************************/

#include "spip/SampleFoldRAM.h"
#include "spip/ContainerRAM.h"

#include <iostream>
#include <cstring>
#include <stdexcept>
#include <cmath>

using namespace std;

spip::SampleFoldRAM::SampleFoldRAM ()
{
}

spip::SampleFoldRAM::~SampleFoldRAM ()
{
}

void spip::SampleFoldRAM::configure (spip::Ordering output_order)
{
  if (verbose)
    cerr << "spip::SampleFoldRAM::configure" << endl;
  if (!buffer)
    buffer = new spip::ContainerRAM();
  spip::SampleFold::configure (output_order);
}

void spip::SampleFoldRAM::transform_ALL_to_TSPFB ()
{
  if (verbose)
    cerr << "spip::SampleFoldRAM::transform_ALL_to_TSPFB" << endl;

  float * in  = (float *) input->get_buffer();
  float * buf = (float *) buffer->get_buffer();
  float * out = (float *) output->get_buffer();

  // index on output data
  for (uint64_t idat=0; idat<ndat; idat++)
  {
    const uint64_t odat    = (idat + dat_offset) / dat_dec;
    const uint64_t out_bin = (idat + dat_offset) % nfolding_bin;

    const uint64_t in_dat_offset  = idat * input->get_dat_stride();
    const uint64_t out_dat_offset = out_bin * output->get_bin_stride(); // buffer has only 1 idat

    for (unsigned isig=0; isig<nsignal; isig++)
    {
      const uint64_t in_sig_offset  = isig * input->get_sig_stride();
      const uint64_t out_sig_offset = isig * output->get_sig_stride();
      for (unsigned ipol=0; ipol<npol; ipol++)
      {
        const uint64_t in_pol_offset  = ipol * input->get_pol_stride();
        const uint64_t out_pol_offset = ipol * output->get_pol_stride();
        for (unsigned ichan=0; ichan<nchan; ichan++)
        {
          const uint64_t in_chan_offset  = ichan * input->get_chan_stride();
          const uint64_t out_chan_offset = ichan * output->get_chan_stride();

          const uint64_t idx = in_dat_offset  + in_sig_offset  + in_pol_offset  + in_chan_offset;
          const uint64_t odx = out_dat_offset + out_sig_offset + out_pol_offset + out_chan_offset;

          for (unsigned idim=0; idim<ndim; idim++)
          {
            buf[odx+idim] += in[idx + idim];
          }
        }
      }
    }

    // increment the number of samples integrated into the buffer
    buffer_idat++;

    // we have completed a time-sample sub-integration
    if (buffer_idat == dat_dec)
    {
      // TODO remove memcpy
      memcpy (out, buf, output->get_dat_stride() * sizeof(float));
      
      buffer->zero();
      buffer_idat = 0;
      out += output->get_dat_stride();
    }
  }

  // ensure the phase offset is maintained
  dat_offset = (dat_offset + ndat) % nfolding_bin;
}


void spip::SampleFoldRAM::transform_SFPT_to_TSPFB ()
{
  if (verbose)
    cerr << "spip::SampleFoldRAM::transform_SFPT_to_TSPFB" << endl;
  transform_ALL_to_TSPFB();
}

void spip::SampleFoldRAM::transform_TSPF_to_TSPFB ()
{
  if (verbose)
    cerr << "spip::SampleFoldRAM::transform_TSPF_to_TSPFB" << endl;
  transform_ALL_to_TSPFB();
}
          
void spip::SampleFoldRAM::transform_TFPS_to_TSPFB ()
{
  if (verbose)
    cerr << "spip::SampleFoldRAM::transform_TFPS_to_TSPFB" << endl;
  transform_ALL_to_TSPFB();
}
