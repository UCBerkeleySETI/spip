//-*-C++-*-
/***************************************************************************
 *
 *   Copyright (C) 2017 by Andrew Jameson
 *   Licensed under the Academic Free License version 2.1
 *
 ***************************************************************************/

#ifndef __BackwardFFTFFTW_h
#define __BackwardFFTFFTW_h

#include "config.h"

#include "spip/BackwardFFT.h"

#include <fftw3.h>

namespace spip {

  class BackwardFFTFFTW: public BackwardFFT
  {
    public:
    
      BackwardFFTFFTW ();
      
      ~BackwardFFTFFTW ();
      
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
