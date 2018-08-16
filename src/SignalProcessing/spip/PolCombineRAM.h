//-*-C++-*-
/***************************************************************************
 *
 *   Copyright (C) 2018 by Andrew Jameson
 *   Licensed under the Academic Free License version 2.1
 *
 ***************************************************************************/

#ifndef __PolCombineRAM_h
#define __PolCombineRAM_h

#include "config.h"

#include "spip/PolCombine.h"
#include "spip/ContainerRAM.h"

namespace spip {

  class PolCombineRAM: public PolCombine
  {
    public:
    
      PolCombineRAM ();
      
      ~PolCombineRAM ();

      void combine_SFPT ();

    protected:
    
    private:

  };
}

#endif
