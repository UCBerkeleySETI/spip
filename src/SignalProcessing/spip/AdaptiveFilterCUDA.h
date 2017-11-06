//-*-C++-*-
/***************************************************************************
 *
 *   Copyright (C) 2017 by Andrew Jameson
 *   Licensed under the Academic Free License version 2.1
 *
 ***************************************************************************/

#ifndef __AdaptiveFilterCUDA_h
#define __AdaptiveFilterCUDA_h

#include "config.h"

#include "spip/AdaptiveFilter.h"

namespace spip {

  class AdaptiveFilterCUDA: public AdaptiveFilter
  {
    public:
    
      AdaptiveFilterCUDA ();
      
      ~AdaptiveFilterCUDA ();
      
      void configure ();

      void prepare ();
      
      void reserve ();
      
      void transform_TSPF ();

    protected:
    
    private:
      
  };
}

#endif
