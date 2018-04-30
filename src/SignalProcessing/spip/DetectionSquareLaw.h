//-*-C++-*-
/***************************************************************************
 *
 *   Copyright (C) 2017 by Andrew Jameson
 *   Licensed under the Academic Free License version 2.1
 *
 ***************************************************************************/

#ifndef __DetectionSquareLaw_h
#define __DetectionSquareLaw_h

#include "spip/Detection.h"

namespace spip {

  class DetectionSquareLaw: public Detection
  {
    public:
     
      DetectionSquareLaw (const char *);

      ~DetectionSquareLaw ();

      void configure (Ordering output_order);

      void set_output_state (Signal::State _state);

      //! Required data transformations
      virtual void transform_SFPT_to_SFPT () = 0;

      virtual void transform_TFPS_to_TFPS () = 0;

      virtual void transform_TSPF_to_TSPF () = 0;

      virtual void transform_TSPFB_to_TSPFB () = 0;

    protected:

    private:

  };

}

#endif
