/***************************************************************************
 *
 *   Copyright (C) 2016 Andrew Jameson
 *   Licensed under the Academic Free License version 2.1
 *
 ***************************************************************************/

#include "spip/PolSelectRAM.h"

#include <stdexcept>
#include <float.h>
#include <string.h>
#include <complex.h>
#include <cmath>

using namespace std;

spip::PolSelectRAM::PolSelectRAM ()
{
}

spip::PolSelectRAM::~PolSelectRAM ()
{
}

void spip::PolSelectRAM::bypass ()
{
  if (verbose)
    cerr << "spip::PolSelectRAM::bypass()" << endl;

  size_t nbytes = input->get_size();
  void * in = (void *) input->get_buffer();
  void * out = (void *) output->get_buffer();

  memcpy (out, in, nbytes);
}

void spip::PolSelectRAM::transform_TSPF()
{
  if (verbose)
    cerr << "spip::PolSelectRAM::transform_TSPF ()" << endl;
}

void spip::PolSelectRAM::transform_SFPT()
{
  if (verbose)
    cerr << "spip::PolSelectRAM::transform_SFPT()" << endl;

  if (verbose)
    cerr << "spip::PolSelectRAM::transform_SFPT npol=" << npol << " out_npol=" << out_npol << endl;

  // pointers to the buffers for in, rfi and out
  float * in = (float *) input->get_buffer();
  float * out = (float *) output->get_buffer();

  // strides through the data block
  const uint64_t pol_stride = ndat * ndim;
  const uint64_t in_chan_stride  = npol * pol_stride;
  const uint64_t out_chan_stride = out_npol * pol_stride;
  const uint64_t in_sig_stride  = nchan * in_chan_stride;
  const uint64_t out_sig_stride = nchan * out_chan_stride;

  // loop over the dimensions of the input block
  for (unsigned isig=0; isig<nsignal; isig++)
  {
    const uint64_t in_sig_offset  = isig * in_sig_stride;
    const uint64_t out_sig_offset = isig * out_sig_stride;
    for (unsigned ichan=0; ichan<nchan; ichan++)
    {
      const uint64_t in_chan_offset = in_sig_offset + ichan * in_chan_stride;
      const uint64_t out_chan_offset = out_sig_offset + ichan * out_chan_stride;
      for (unsigned ipol=0; ipol<out_npol; ipol++)
      {
        const uint64_t in_sfp_offset = in_chan_offset + ipol * pol_stride;
        const uint64_t out_sfp_offset = out_chan_offset + ipol * pol_stride;

        if (verbose)
          cerr << "spip::PolSelectRAM::transform_SFPT ipol=" << ipol << " in+=" << in_sfp_offset << " out_sfp_offset=" << out_sfp_offset << endl;

        // offset pointers for this signal, channel and polarisation
        void * in_ptr  = (void *) (in + in_sfp_offset);
        void * out_ptr = (void *) (out + out_sfp_offset);
        size_t nbytes = pol_stride * sizeof (float);

        if (verbose)
          cerr << "spip::PolSelectRAM::transform_SFPT memcpy (" << out_ptr << ", " << in_ptr << ", " << nbytes << ")" << endl;
        memcpy (out_ptr, in_ptr, nbytes);
      }
    }
  }
  if (verbose)
    cerr << "spip::PolSelectRAM::transform_SFPT done" << endl;
}
