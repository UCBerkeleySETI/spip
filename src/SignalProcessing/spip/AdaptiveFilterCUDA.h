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
    
      AdaptiveFilterCUDA ();
      
      ~AdaptiveFilterCUDA ();

      void set_input_rfi (Container *);
      
      void configure ();

      void prepare ();
      
      void reserve ();
      
      void transform_TSPF ();

      void transform_SFPT ();

    protected:
    
    private:

      // the current gain for each channel and polarisation
      ContainerCUDADevice * gains;
      
  };
}

#endif
