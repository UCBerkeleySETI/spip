//-*-C++-*-
/***************************************************************************
 *
 *   Copyright (C) 2018 by Andrew Jameson
 *   Licensed under the Academic Free License version 2.1
 *
 ***************************************************************************/

#ifndef __PolSelectCUDA_h
#define __PolSelectCUDA_h

#include "config.h"

#include "spip/PolSelect.h"
#include "spip/ContainerCUDA.h"
#include <cuda_runtime.h>

namespace spip {

  class PolSelectCUDA: public PolSelect
  {
    public:
    
      PolSelectCUDA (cudaStream_t);
      
      ~PolSelectCUDA ();

      void transform_TSPF ();

      void transform_SFPT ();

    protected:
    
    private:

      cudaStream_t stream;

  };
}

#endif
