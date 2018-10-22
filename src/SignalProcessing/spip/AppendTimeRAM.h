//-*-C++-*-
/***************************************************************************
 *
 *   Copyright (C) 2018 by Andrew Jameson
 *   Licensed under the Academic Free License version 2.1
 *
 ***************************************************************************/

#ifndef __AppendTimeRAM_h
#define __AppendTimeRAM_h

#include "spip/AppendTime.h"

#include <cstring>

namespace spip {

  class AppendTimeRAM: public AppendTime
  {
    public:
    
      AppendTimeRAM ();
      
      ~AppendTimeRAM ();
 
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
            for (unsigned ipol=0; ipol<npol; ipol++)
            {
              uint64_t dat_offset = 0;
              for (unsigned i=0;i<inputs.size(); i++)
              {
                // size of data to copy from input i
                const uint64_t in_stride = inputs[i]->get_pol_stride() * sizeof (T);

                // offset for the input
                const uint64_t in_offset = isig * inputs[i]->get_sig_stride() + 
                                           ichan * inputs[i]->get_chan_stride() + 
                                           ipol * inputs[i]->get_pol_stride();

                const uint64_t out_offset = isig * output->get_sig_stride() + 
                                            ichan * output->get_chan_stride() + 
                                            ipol * output->get_pol_stride() +
                                            dat_offset;
 
                // copy from input[i] to output
                T * in  = ((T *) inputs[i]->get_buffer()) + in_offset;
                T * out = ((T *) output->get_buffer()) + out_offset;
                memcpy ((void *) out, (void *) in, in_stride);

                dat_offset += in_stride;
              }
            }
          }
        }
      }

      template <class T>
      void combine_tfps_to_tfps (T dummy)
      {
        T * out = (T *) output->get_buffer();

        for (unsigned i=0;i<inputs.size(); i++)
        {
          T * in  = ((T *) inputs[i]->get_buffer());

          // size of data to copy from inputs[i]
          const uint64_t input_ndat = inputs[i]->get_ndat();
          const uint64_t input_nval = input_ndat * inputs[i]->get_dat_stride();
          const uint64_t in_stride = input_nval * sizeof(T);

          memcpy ((void *) out, (void *) in, in_stride);

          out += input_nval;
        }
      }

      void combine_TSPF_to_TSPF ();

      void combine_TSPFB_to_TSPFB ();

    protected:
    
    private:

  };
}

#endif
