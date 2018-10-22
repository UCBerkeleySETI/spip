//-*-C++-*-
/***************************************************************************
 *
 *   Copyright (C) 2018 by Andrew Jameson
 *   Licensed under the Academic Free License version 2.1
 *
 ***************************************************************************/

#ifndef __ReverseFrequencyRAM_h
#define __ReverseFrequencyRAM_h

#include "config.h"

#include "spip/ReverseFrequency.h"
#include "spip/ContainerRAM.h"

namespace spip {

  class ReverseFrequencyRAM: public ReverseFrequency
  {
    public:
    
      ReverseFrequencyRAM ();
      
      ~ReverseFrequencyRAM ();
 
      void transform_TSPF_to_TSPF ();

      void transform_TFPS_to_TFPS ();

      void transform_copy ();
      
    protected:
    
    private:

  };
}

#endif
