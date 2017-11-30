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
      
      void configure (Ordering output_order);

      void configure_plan ();

      void prepare ();
      
      void reserve ();
      
      void transform_SFPT_to_TFPS ();

      void transform_SFPT_to_TSPF ();

      void transform_SFPT_to_SFPT ();

    protected:
    
    private:
    
      fftwf_plan plan;

  };
}

#endif
