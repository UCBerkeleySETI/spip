//-*-C++-*-
/***************************************************************************
 *
 *   Copyright (C) 2017 by Andrew Jameson
 *   Licensed under the Academic Free License version 2.1
 *
 ***************************************************************************/

#ifndef __AdaptiveFilterCUDA_h
#define __AdaptiveFilterCUDA_h

#include "config.h"

#include "spip/AdaptiveFilter.h"
#include "spip/ContainerCUDA.h"
#include <cuda_runtime.h>

namespace spip {

  class AdaptiveFilterCUDA: public AdaptiveFilter
  {
    public:
    
      AdaptiveFilterCUDA (cudaStream_t, std::string);
      
      ~AdaptiveFilterCUDA ();

      void configure (Ordering output_order);

      void transform_TSPF ();

      void transform_SFPT ();

      void write_gains ();

    protected:
    
    private:

      cudaStream_t stream;
  };
}

#endif
