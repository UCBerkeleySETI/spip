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
#include <climits>

namespace spip {

  class UnpackFloatRAM: public UnpackFloat
  {
    public:
    
      UnpackFloatRAM ();
      
      ~UnpackFloatRAM ();
 
      void prepare ();
      
      void reserve ();
   
      void transform_SFPT_to_SFPT ();
      
    protected:
    
    private:

      inline float    convert_twos (float in)    { return in; };
      inline uint16_t convert_twos (uint16_t in) { return in ^ 0x8000; };
      inline int16_t  convert_twos (int16_t in)  { return in ^ 0x8000; };
      inline int8_t   convert_twos (int8_t in)   { return in ^ 0x80; };
      inline uint8_t  convert_twos (uint8_t in)  { return in ^ 0x80; };

      template <class T>
      void unpack_sfpt_to_sfpt (T * in, float * out)
      {
        uint64_t nval = ndim * ndat * npol * nchan * nsignal;
        for (uint64_t ival=0; ival < nval; ival++)
        {
          T raw = in[ival]; 

          // first convert Endian if required
          if (endianness != spip::Endian::Little)
            raw = swap_endian(raw);

          if (encoding != spip::Encoding::TwosComplement)
            raw = convert_twos(raw);

          out[ival] = (float(raw) + offset) * scale;
        }
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
                if (endianness != spip::Endian::Little)
                {
                  out[signal_offset] = (float(swap_endian(*in)) + offset) * scale;
                }
                else
                {
                  out[signal_offset] = (float(*in) + offset) * scale;
                }
                in++;
              }
            }
          }
        }
      }

      template <typename T>
      T swap_endian(T u)
      {
        static_assert (CHAR_BIT == 8, "CHAR_BIT != 8");
        union
        {
          T u;
          unsigned char u8[sizeof(T)];
        } source, dest;

        source.u = u;

        for (size_t k = 0; k < sizeof(T); k++)
          dest.u8[k] = source.u8[sizeof(T) - k - 1];

        return dest.u;
      }

  };
}

#endif
