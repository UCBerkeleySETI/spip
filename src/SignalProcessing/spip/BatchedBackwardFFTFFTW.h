//-*-C++-*-
/***************************************************************************
 *
 *   Copyright (C) 2017 by Andrew Jameson
 *   Licensed under the Academic Free License version 2.1
 *
 ***************************************************************************/

#ifndef __BatchedBackwardFFTFFTW_h
#define __BatchedBackwardFFTFFTW_h

#include "config.h"

#include "spip/BatchedBackwardFFT.h"

#include <fftw3.h>

namespace spip {

  class BatchedBackwardFFTFFTW: public BatchedBackwardFFT
  {
    public:
    
      BatchedBackwardFFTFFTW ();
      
      ~BatchedBackwardFFTFFTW ();
 
      void configure ();

      void prepare ();
      
      void reserve ();
      
      void transform ();
      
    protected:
    
    private:
    
      fftwf_plan plan;
      
  };
}

#endif
