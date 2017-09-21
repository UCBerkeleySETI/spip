//-*-C++-*-
/***************************************************************************
 *
 *   Copyright (C) 2017 by Andrew Jameson
 *   Licensed under the Academic Free License version 2.1
 *
 ***************************************************************************/

#ifndef __ForwardFFTFFTW_h
#define __ForwardFFTFFTW_h

#include "config.h"

#include "spip/ForwardFFT.h"

#include <fftw3.h>

namespace spip {

  class ForwardFFTFFTW: public ForwardFFT
  {
    public:
    
      ForwardFFTFFTW ();
      
      ~ForwardFFTFFTW ();
      
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