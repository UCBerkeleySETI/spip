//-*-C++-*-
/***************************************************************************
 *
 *   Copyright (C) 2018 by Andrew Jameson
 *   Licensed under the Academic Free License version 2.1
 *
 ***************************************************************************/

#ifndef __ReverseFrequencyCUDA_h
#define __ReverseFrequencyCUDA_h

#include "config.h"

#include "spip/ContainerCUDA.h"
#include "spip/ReverseFrequency.h"

#include <cuda_runtime.h>

namespace spip {

  class ReverseFrequencyCUDA: public ReverseFrequency
  {
    public:
    
      ReverseFrequencyCUDA (cudaStream_t);
      
      ~ReverseFrequencyCUDA ();
 
      void transform_SFPT_to_SFPT ();

      void transform_TSPF_to_TSPF ();

      void transform_TFPS_to_TFPS ();

      void transform_copy();
      
    protected:
    
    private:

      cudaStream_t stream;

  };
}

#endif
