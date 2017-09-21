//-*-C++-*-
/***************************************************************************
 *
 *   Copyright (C) 2017 by Andrew Jameson
 *   Licensed under the Academic Free License version 2.1
 *
 ***************************************************************************/

#ifndef __ForwardFFTCUDA_h
#define __ForwardFFTCUDA_h

#include "config.h"

#include "spip/ForwardFFT.h"

#include <cuda_runtime.h>
#include <cufft.h>

namespace spip {

  class ForwardFFTCUDA: public ForwardFFT
  {
    public:
    
      ForwardFFTCUDA (cudaStream_t);
      
      ~ForwardFFTCUDA ();
      
      void prepare ();

      void prepare_plan (uint64_t ndat);
      
      void reserve ();
      
      void transform ();
      
    protected:
    
    private:
 
      cudaStream_t stream;

      cufftHandle plan;

      unsigned nbatch;

      void * work_area;

      size_t work_area_size;

      bool auto_allocate;

  };
}

#endif
