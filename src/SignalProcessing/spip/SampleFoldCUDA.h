//-*-C++-*-
/***************************************************************************
 *
 *   Copyright (C) 2018 by Andrew Jameson
 *   Licensed under the Academic Free License version 2.1
 *
 ***************************************************************************/

#ifndef __SampleFoldCUDA_h
#define __SampleFoldCUDA_h

#include "config.h"

#include "spip/SampleFold.h"

namespace spip {

  class SampleFoldCUDA: public SampleFold
  {
    public:
    
      SampleFoldCUDA (cudaStream_t stream);
      
      ~SampleFoldCUDA ();
 
      void configure (spip::Ordering);
      
      void transform_SFPT_to_TSPFB ();

      void transform_TFPS_to_TSPFB ();

      void transform_TSPF_to_TSPFB ();
      
    protected:
    
    private:

      cudaStream_t stream;
  };
}

#endif
