//-*-C++-*-
/***************************************************************************
 *
 *   Copyright (C) 2018 by Andrew Jameson
 *   Licensed under the Academic Free License version 2.1
 *
 ***************************************************************************/

#ifndef __DetectionPolarimetryRAM_h
#define __DetectionPolarimetryRAM_h

#include "config.h"

#include "spip/DetectionPolarimetry.h"

namespace spip {

  class DetectionPolarimetryRAM: public DetectionPolarimetry
  {
    public:
    
      DetectionPolarimetryRAM ();
      
      ~DetectionPolarimetryRAM ();
 
      void reserve ();
   
      void transform_SFPT_to_SFPT ();

      void transform_TSPF_to_TSPF ();

      void transform_TFPS_to_TFPS ();

    protected:
    
    private:

      void cross_detect (float, float, float, float, float *, float *, float *, float *);

      void stokes_detect (float, float, float, float, float *, float *, float *, float *);

  };
}

#endif
