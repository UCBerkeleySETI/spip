//-*-C++-*-
/***************************************************************************
 *
 *   Copyright (C) 2018 by Andrew Jameson
 *   Licensed under the Academic Free License version 2.1
 *
 ***************************************************************************/

#ifndef __AppendFrequencyRAM_h
#define __AppendFrequencyRAM_h

#include "spip/AppendFrequency.h"

#include <cstring>

namespace spip {

  class AppendFrequencyRAM: public AppendFrequency
  {
    public:
    
      AppendFrequencyRAM ();
      
      ~AppendFrequencyRAM ();
 
      void reserve ();

      void combine_SFPT_to_SFPT();

      void combine_TFPS_to_TFPS();
 
      template <class T>
      void combine_sfpt_to_sfpt (T dummy)
      {
        for (unsigned isig=0; isig<nsignal; isig++)
        {
          for (unsigned ichan=0; ichan<nchan; ichan++)
          {
            const uint64_t input_chan_offset = 0;
            for (unsigned i=0;i<inputs.size(); i++)
            {
              const uint64_t in_stride = inputs[i]->get_pol_stride() * sizeof (T);
              const uint64_t in_offset = isig * inputs[i]->get_sig_stride() + ichan * inputs[i]->get_chan_stride();
              const uint64_t out_offset = isig * output->get_sig_stride() + ichan * output->get_chan_stride() + input_chan_offset;

              T * in  = ((T *) inputs[i]->get_buffer()) + in_offset;
              T * out = ((T *) output->get_buffer()) + out_offset;

              memcpy ((void *) out, (void *) in, in_stride);

              input_chan_offset += in_stride;
            }
          }
        }
      }

      template <class T>
      void combine_tfps_to_tfps (T dummy)
      {
        T * out = (T *) output->get_buffer();
        uint64_t input_chan_offset = 0;
        for (unsigned i=0;i<inputs.size(); i++)
        {
          T * in  = ((T *) inputs[i]->get_buffer());
          for (uint64_t idat=0; idat<ndat; idat++)
          {
            const uint64_t in_dat_offset = idat * inputs[i]->get_dat_stride();
            const uint64_t out_dat_offset = idat * output->get_dat_stride();
            for (unsigned ichan=0; ichan<nchan; ichan++)
            {
              const uint64_t in_chan_offset = in_dat_offset + ichan * inputs[i]->get_chan_stride();
              const uint64_t out_chan_offset = out_dat_offset + ichan * output->get_chan_stride() + input_chan_offset;
              for (unsigned ipol=0; ipol<npol; ipol++)
              {
                const uint64_t in_pol_offset = in_chan_offset + ipol * inputs[i]->get_pol_stride();
                const uint64_t out_pol_offset = out_chan_offset + ipol * output->get_pol_stride();
                for (unsigned isig=0; isig<nsignal; isig++)
                {
                  const uint64_t in_sig_offset = in_pol_offset + isig * inputs[i]->get_sig_stride();
                  const uint64_t out_sig_offset = out_pol_offset + isig * output->get_sig_stride();
                  
                  out[out_sig_offset] = in[in_sig_offset];
                }
              }
            } 
          }
          input_chan_offset = inputs[i]->get_chan_stride();
        }
      }

      void combine_TSPF_to_TSPF ();

      void combine_TSPFB_to_TSPFB ();

    protected:
    
    private:

  };
}

#endif
