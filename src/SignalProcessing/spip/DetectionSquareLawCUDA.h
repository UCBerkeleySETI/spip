//-*-C++-*-
/***************************************************************************
 *
 *   Copyright (C) 2018 by Andrew Jameson
 *   Licensed under the Academic Free License version 2.1
 *
 ***************************************************************************/

#ifndef __DetectionSquareLawCUDA_h
#define __DetectionSquareLawCUDA_h

#include "config.h"

#include "spip/DetectionSquareLaw.h"

#include <cuda_runtime.h>

namespace spip {

  class DetectionSquareLawCUDA: public DetectionSquareLaw
  {
    public:
    
      DetectionSquareLawCUDA (cudaStream_t);
      
      ~DetectionSquareLawCUDA ();
 
      void reserve ();
   
      void transform_SFPT_to_SFPT ();

      void transform_TFPS_to_TFPS ();
      
      void transform_TSPF_to_TSPF ();

      void transform_TSPFB_to_TSPFB ();

    protected:
    
    private:

      cudaStream_t stream;

  };
}

#endif
