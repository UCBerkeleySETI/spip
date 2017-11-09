//-*-C++-*-
/***************************************************************************
 *
 *   Copyright (C) 2017 by Andrew Jameson
 *   Licensed under the Academic Free License version 2.1
 *
 ***************************************************************************/

#ifndef __UnpackFloatCUDA_h
#define __UnpackFloatCUDA_h

#include "config.h"

#include "spip/UnpackFloat.h"

#include <iostream>
#include <cuda_runtime.h>

namespace spip {

  class UnpackFloatCUDA: public UnpackFloat
  {
    public:
    
      UnpackFloatCUDA (cudaStream_t);
      
      ~UnpackFloatCUDA ();
 
      void configure();

      void prepare ();
      
      void reserve ();
      
      void transform_SFPT_to_SFPT ();
      
    protected:
    
    private:

      cudaStream_t stream;

  };
}

#endif
