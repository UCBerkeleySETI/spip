//-*-C++-*-
/***************************************************************************
 *
 *   Copyright (C) 2018 by Andrew Jameson
 *   Licensed under the Academic Free License version 2.1
 *
 ***************************************************************************/

#ifndef __DetectionSquareLawRAM_h
#define __DetectionSquareLawRAM_h

#include "config.h"

#include "spip/DetectionSquareLaw.h"

namespace spip {

  class DetectionSquareLawRAM: public DetectionSquareLaw
  {
    public:
    
      DetectionSquareLawRAM ();
      
      ~DetectionSquareLawRAM ();
 
      void reserve ();
   
      void transform_SFPT_to_SFPT ();

      void transform_TFPS_to_TFPS ();

      void transform_TSPF_to_TSPF ();

      void transform_TSPFB_to_TSPFB ();

    protected:
    
    private:

  };
}

#endif
