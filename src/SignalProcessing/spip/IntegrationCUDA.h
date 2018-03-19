//-*-C++-*-
/***************************************************************************
 *
 *   Copyright (C) 2018 by Andrew Jameson
 *   Licensed under the Academic Free License version 2.1
 *
 ***************************************************************************/

#ifndef __IntegrationCUDA_h
#define __IntegrationCUDA_h

#include "config.h"

#include "spip/ContainerCUDA.h"
#include "spip/Integration.h"

#include <cuda_runtime.h>

namespace spip {

  class IntegrationCUDA: public Integration
  {
    public:
    
      IntegrationCUDA (cudaStream_t);
      
      ~IntegrationCUDA ();
 
      void configure (spip::Ordering output_order);

      void reserve ();
   
      void transform_SFPT_to_SFPT ();

      void transform_TSPF_to_TSPF ();

      void transform_TFPS_to_TFPS ();
      
    protected:
    
    private:

      cudaStream_t stream;

      ContainerCUDADevice * fscrunched;

  };
}

#endif
