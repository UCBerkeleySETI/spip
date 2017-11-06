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

      void configure_plan ();

      void prepare ();
      
      void reserve ();
      
      void transform_TFPS_to_SFPT ();

      void transform_TSPF_to_SFPT ();

    protected:
    
    private:
    
      fftwf_plan plan;
      
  };
}

#endif
