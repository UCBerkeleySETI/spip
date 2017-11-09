//-*-C++-*-
/***************************************************************************
 *
 *   Copyright (C) 2017 by Andrew Jameson
 *   Licensed under the Academic Free License version 2.1
 *
 ***************************************************************************/

#ifndef __UnpackFloatRAM_h
#define __UnpackFloatRAM_h

#include "config.h"

#include "spip/UnpackFloat.h"

#include <iostream>

namespace spip {

  class UnpackFloatRAM: public UnpackFloat
  {
    public:
    
      UnpackFloatRAM ();
      
      ~UnpackFloatRAM ();
 
      void configure();

      void prepare ();
      
      void reserve ();
   
      void transform_SFPT_to_SFPT ();
      
    protected:
    
    private:

      template <class T>
      void unpack_sfpt_to_sfpt (T * in, float * out)
      {
        uint64_t nval = ndim * ndat * npol * nchan * nsignal;
        for (uint64_t ival=0; ival < nval; ival++)
          out[ival] = float(in[ival]);
      }

      template <class T>
      void unpack_tfps_to_tfps (T * in, float * out)
      {
        // output strides
        const uint64_t pol_stride = ndat; 
        const uint64_t chan_stride = npol * pol_stride;
        const uint64_t signal_stride  = nchan * chan_stride;
        
        for (uint64_t idat=0; idat<ndat; idat++)
        { 
          const uint64_t samp_offset = idat;
          for (unsigned ichan=0; ichan<nchan; ichan++)
          { 
            const uint64_t chan_offset = samp_offset + ichan * chan_stride;
            for (unsigned ipol=0; ipol<npol; ipol++)
            { 
              const uint64_t pol_offset = chan_offset + ipol * pol_stride;
              for (unsigned isig=0; isig<nsignal; isig++)
              { 
                const uint64_t signal_offset = pol_offset + isig * signal_stride;
                out[signal_offset] = float(*in);
                in++;
              }
            }
          }
        }
      }

  };
}

#endif
