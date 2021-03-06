//-*-C++-*-
/***************************************************************************
 *
 *   Copyright (C) 2018 by Andrew Jameson
 *   Licensed under the Academic Free License version 2.1
 *
 ***************************************************************************/

#ifndef __DetectionPolarimetryCUDA_h
#define __DetectionPolarimetryCUDA_h

#include "config.h"

#include "spip/DetectionPolarimetry.h"

#include <cuda_runtime.h>

namespace spip {

  class DetectionPolarimetryCUDA: public DetectionPolarimetry
  {
    public:
    
      DetectionPolarimetryCUDA (cudaStream_t stream);
      
      ~DetectionPolarimetryCUDA ();
 
      void reserve ();
   
      void transform_SFPT_to_SFPT ();

      void transform_SFPT_to_TSPF ();

      void transform_TSPF_to_TSPF ();

      void transform_TFPS_to_TFPS ();

      void transform_TSPFB_to_TSPFB ();

    protected:
    
    private:

      void cross_detect (float, float, float, float, float *, float *, float *, float *);

      void stokes_detect (float, float, float, float, float *, float *, float *, float *);

      cudaStream_t stream;

  };
}

#endif
