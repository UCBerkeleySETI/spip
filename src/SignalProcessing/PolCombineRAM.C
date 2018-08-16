/***************************************************************************
 *
 *   Copyright (C) 2016 Andrew Jameson
 *   Licensed under the Academic Free License version 2.1
 *
 ***************************************************************************/

#include "spip/PolCombineRAM.h"

#include <stdexcept>
#include <float.h>
#include <string.h>
#include <complex.h>
#include <cmath>

using namespace std;

spip::PolCombineRAM::PolCombineRAM ()
{
}

spip::PolCombineRAM::~PolCombineRAM ()
{
}

void spip::PolCombineRAM::combine_SFPT()
{
  if (verbose)
    cerr << "spip::PolCombineRAM::combine_SFPT()" << endl;

  if (verbose)
    cerr << "spip::PolCombineRAM::combine_SFPT total_npol=" << total_npol << endl;

  float * out = (float *) output->get_buffer();
  unsigned int opol = 0;
  
  for (unsigned i=0;i<inputs.size(); i++)
  {
    float * in = (float *) inputs[i]->get_buffer();
    const unsigned npol = inputs[i]->get_npol();
    for (unsigned isig=0; isig<nsignal; isig++)
    {
      const uint64_t in_sig_offset  = isig * inputs[i]->get_sig_stride();
      const uint64_t out_sig_offset = isig * output->get_sig_stride();
      for (unsigned ichan=0; ichan<nchan; ichan++)
      {
        const uint64_t in_chan_offset = in_sig_offset + ichan * inputs[i]->get_chan_stride();
        const uint64_t out_chan_offset = out_sig_offset + ichan * output->get_chan_stride();
        for (unsigned ipol=0; ipol<npol; ipol++)
        {
          const uint64_t in_sfp_offset = in_chan_offset + ipol * inputs[i]->get_pol_stride();
          const uint64_t out_sfp_offset = out_chan_offset + (opol + ipol) * output->get_pol_stride();

          if (verbose)
            cerr << "spip::PolCombineRAM::combine_SFPT ipol=" 
                 << ipol << " in+=" << in_sfp_offset 
                 << " out_sfp_offset=" << out_sfp_offset << endl;

          // offset pointers for this signal, channel and polarisation
          void * in_ptr  = (void *) (in + in_sfp_offset);
          void * out_ptr = (void *) (out + out_sfp_offset);
          size_t nbytes = inputs[i]->get_pol_stride() * sizeof (float);

          if (verbose)
            cerr << "spip::PolCombineRAM::combine_SFPT memcpy (" 
                 << out_ptr << ", " << in_ptr << ", " << nbytes << ")" << endl;
          memcpy (out_ptr, in_ptr, nbytes);
        }
      }
    }
    opol += inputs[i]->get_npol();
  }

  if (verbose)
    cerr << "spip::PolCombineRAM::combine_SFPT done" << endl;
}
