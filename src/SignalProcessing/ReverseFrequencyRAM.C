/***************************************************************************
 *
 *   Copyright (C) 2018 Andrew Jameson
 *   Licensed under the Academic Free License version 2.1
 *
 ***************************************************************************/

#include "spip/ReverseFrequencyRAM.h"

#include <iostream>
#include <cstring>
#include <stdexcept>
#include <cmath>

using namespace std;

spip::ReverseFrequencyRAM::ReverseFrequencyRAM ()
{
}

spip::ReverseFrequencyRAM::~ReverseFrequencyRAM ()
{
}

void spip::ReverseFrequencyRAM::transform_TSPF_to_TSPF ()
{
  if (verbose)
    cerr << "spip::ReverseFrequencyRAM::transform_TSPF_to_TSPF" << endl;
  
  if (!reversal)
  {
    transform_copy();
  }
  else
  {
    float * in  = (float *) input->get_buffer();
    float * out = (float *) output->get_buffer();

    uint64_t idx = 0;
    for (uint64_t idat=0; idat<ndat; idat++)
    {
      for (unsigned isig=0; isig<nsignal; isig++)
      { 
        for (unsigned ipol=0; ipol<npol; ipol++)
        { 
          for (unsigned ichan=0; ichan<nchan; ichan++)
          {
            const uint64_t ochan = (nchan - ichan) -1;
            out[idx + ochan] = in[idx + ichan];
          }
          idx += nchan;
        }
      }
    }
  }
}

void spip::ReverseFrequencyRAM::transform_TFPS_to_TFPS ()
{
  if (verbose)
    cerr << "spip::ReverseFrequencyRAM::transform_TFPS_to_TFPS" << endl;

  if (!reversal)
  {
    transform_copy();
  }
  else
  {
    float * in  = (float *) input->get_buffer();
    float * out = (float *) output->get_buffer();

    const uint64_t dat_stride = output->get_dat_stride();
    const uint64_t chan_stride = output->get_chan_stride();
    const uint64_t pol_stride = output->get_pol_stride();

    uint64_t idx = 0;

    for (uint64_t idat=0; idat<ndat; idat++)
    {
      const uint64_t out_dat_offset = idat * dat_stride;
      for (unsigned ichan=0; ichan<nchan; ichan++)
      {
        const unsigned ochan = (nchan - ichan) - 1;
        const uint64_t out_chan_offset = out_dat_offset + (ochan * chan_stride);
        for (unsigned ipol=0; ipol<npol; ipol++)
        {
          const uint64_t out_pol_offset = out_chan_offset + (ipol * pol_stride);
          for (unsigned isig=0; isig<nsignal; isig++)
          {
            out[out_pol_offset + isig] = in[idx];
            idx++;
          }
        }
      }
    }
  }
}

void spip::ReverseFrequencyRAM::transform_copy ()
{
  if (verbose)
    cerr << "spip::ReverseFrequencyRAM::transform_copy" << endl;

  void * in = (void *) input->get_buffer();
  void * out = (void *) output->get_buffer();

  if (this->get_type() == Behaviour::inplace)
  {
    if (verbose)
      cerr << "spip::ReverseFrequencyRAM::transform_copy no operation required "
           << " due to in-place and no reversal" << endl;
  }
  else
  {
    memcpy (out, in, input->get_size());
  }
}
