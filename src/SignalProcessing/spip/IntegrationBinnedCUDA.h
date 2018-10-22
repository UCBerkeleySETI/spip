//-*-C++-*-
/***************************************************************************
 *
 *   Copyright (C) 2018 by Andrew Jameson
 *   Licensed under the Academic Free License version 2.1
 *
 ***************************************************************************/

#ifndef __IntegrationBinnedCUDA_h
#define __IntegrationBinnedCUDA_h

#include "config.h"

#include "spip/ContainerCUDA.h"
#include "spip/IntegrationBinned.h"

#include <cuda_runtime.h>

namespace spip {

  class IntegrationBinnedCUDA: public IntegrationBinned
  {
    public:
    
      IntegrationBinnedCUDA (cudaStream_t);
      
      ~IntegrationBinnedCUDA ();
 
      void configure (spip::Ordering output_order);

      void prepare_binplan ();

      void transform_TSPF_to_TSPFB ();

    protected:
    
    private:

      cudaStream_t stream;

      ContainerCUDADevice * fscrunched;

  };
}

#endif
