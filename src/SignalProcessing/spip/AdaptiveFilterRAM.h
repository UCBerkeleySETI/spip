//-*-C++-*-
/***************************************************************************
 *
 *   Copyright (C) 2017 by Andrew Jameson
 *   Licensed under the Academic Free License version 2.1
 *
 ***************************************************************************/

#ifndef __AdaptiveFilterRAM_h
#define __AdaptiveFilterRAM_h

#include "config.h"

#include "spip/AdaptiveFilter.h"
#include "spip/ContainerRAM.h"

//#include <float.h>
//#include <complex.h>

namespace spip {

  class AdaptiveFilterRAM: public AdaptiveFilter
  {
    public:
    
      AdaptiveFilterRAM ();
      
      ~AdaptiveFilterRAM ();

      void set_input_rfi (Container *);
      
      void configure ();

      void prepare ();
      
      void reserve ();
      
      void transform_TSPF ();

      void transform_SFPT ();

    protected:
    
    private:

  };
}

#endif
