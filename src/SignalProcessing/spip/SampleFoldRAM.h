//-*-C++-*-
/***************************************************************************
 *
 *   Copyright (C) 2018 by Andrew Jameson
 *   Licensed under the Academic Free License version 2.1
 *
 ***************************************************************************/

#ifndef __SampleFoldRAM_h
#define __SampleFoldRAM_h

#include "config.h"

#include "spip/SampleFold.h"

namespace spip {

  class SampleFoldRAM: public SampleFold
  {
    public:
    
      SampleFoldRAM ();
      
      ~SampleFoldRAM ();
 
      void configure (spip::Ordering);
      
      void transform_ALL_to_TSPFB ();

      void transform_SFPT_to_TSPFB ();

      void transform_TFPS_to_TSPFB ();

      void transform_TSPF_to_TSPFB ();
      
    protected:
    
    private:

  };
}

#endif
