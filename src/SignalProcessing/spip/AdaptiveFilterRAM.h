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

namespace spip {

  class AdaptiveFilterRAM: public AdaptiveFilter
  {
    public:
    
      AdaptiveFilterRAM ();
      
      ~AdaptiveFilterRAM ();
      
      void configure ();

      void prepare ();
      
      void reserve ();
      
      void transform_TSPF ();

    protected:
    
    private:
      
  };
}

#endif
