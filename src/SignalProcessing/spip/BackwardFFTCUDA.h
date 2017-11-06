//-*-C++-*-
/***************************************************************************
 *
 *   Copyright (C) 2017 by Andrew Jameson
 *   Licensed under the Academic Free License version 2.1
 *
 ***************************************************************************/

#ifndef __BackwardFFTCUDA_h
#define __BackwardFFTCUDA_h

#include "config.h"

#include "spip/BackwardFFT.h"

#include <cuda_runtime.h>
#include <cufft.h>

namespace spip {

  class BackwardFFTCUDA: public BackwardFFT
  {
    public:
    
      BackwardFFTCUDA (cudaStream_t);
      
      ~BackwardFFTCUDA ();
      
      void configure ();

      void configure_plan ();

      void prepare ();
      
      void reserve ();
      
      void transform_TFPS_to_SFPT ();

      void transform_TSPF_to_SFPT ();

    protected:
    
    private:

      cudaStream_t stream;

      cufftHandle plan;

      void * work_area;

      size_t work_area_size;

      bool auto_allocate;
  
      
  };
}

#endif
