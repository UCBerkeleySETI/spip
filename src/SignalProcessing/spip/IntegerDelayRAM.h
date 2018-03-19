//-*-C++-*-
/***************************************************************************
 *
 *   Copyright (C) 2016 by Andrew Jameson
 *   Licensed under the Academic Free License version 2.1
 *
 ***************************************************************************/

#ifndef __IntegerDelayRAM_h
#define __IntegerDelayRAM_h

#include "spip/ContainerRAM.h"
#include "spip/IntegerDelay.h"

namespace spip {

  class IntegerDelayRAM: public IntegerDelay
  {
    public:
     
      IntegerDelayRAM ();

      ~IntegerDelayRAM ();

      void prepare ();

      void prepare_output ();

      void transform_SFPT_to_SFPT ();

    private:

      template <typename T>
      void transform (T * in, T* buf, T * out)
      {
        int * delta = (int *) delta_delays->get_buffer();
        uint64_t idat, odat;

        for (unsigned ichan=0; ichan<nchan; ichan++)
        {
          for (unsigned ipol=0; ipol<npol; ipol++)
          {
            // TODO fix the integer transition

            for (unsigned isig=0; isig<nsignal; isig++)
            {
              int delay = delta[isig];
              // copy buffered data to the output
              if (have_buffered_output)
              {
                for (odat=ndat-delay,idat=0; idat<ndat; idat++,odat++)
                  out[odat] = buf[idat];
              }

              // copy input to the buffer
              for (odat=0,idat=delay; idat<ndat; idat++,odat++)
                buf[odat] = in[idat];

              in += ndat;
              out += ndat;

            }
          }
        }
      }

    protected:

    private:

  };

}

#endif
