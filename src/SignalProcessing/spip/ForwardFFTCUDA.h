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
      
      void configure (Ordering);

      void configure_plan ();
      
      void condition ();
      //void fft_shift ();

      void transform_SFPT_to_TFPS ();

      void transform_SFPT_to_TSPF ();

      void transform_SFPT_to_SFPT ();

      //void normalize_output ();

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
