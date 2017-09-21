//-*-C++-*-
/***************************************************************************
 *
 *   Copyright (C) 2017 by Andrew Jameson
 *   Licensed under the Academic Free License version 2.1
 *
 ***************************************************************************/

#ifndef __BatchedBackwardFFTCUDA_h
#define __BatchedBackwardFFTCUDA_h

#include "config.h"

#include "spip/BatchedBackwardFFT.h"

#include <cuda_runtime.h>
#include <cufft.h>

namespace spip {

  class BatchedBackwardFFTCUDA: public BatchedBackwardFFT
  {
    public:
    
      BatchedBackwardFFTCUDA (cudaStream_t);
      
      ~BatchedBackwardFFTCUDA ();
      
      void prepare ();

      void prepare_plan (uint64_t ndat);
      
      void reserve ();
      
      void transform ();
      
    protected:
    
    private:
 
      cudaStream_t stream;

      cufftHandle plan;

      void * work_area;
        
      size_t work_area_size;

      bool auto_allocate;

      unsigned nbatch;
      
  };
}

#endif
