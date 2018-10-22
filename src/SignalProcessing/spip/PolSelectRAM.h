//-*-C++-*-
/***************************************************************************
 *
 *   Copyright (C) 2018 by Andrew Jameson
 *   Licensed under the Academic Free License version 2.1
 *
 ***************************************************************************/

#ifndef __PolSelectRAM_h
#define __PolSelectRAM_h

#include "config.h"

#include "spip/PolSelect.h"
#include "spip/ContainerRAM.h"

namespace spip {

  class PolSelectRAM: public PolSelect
  {
    public:
    
      PolSelectRAM ();
      
      ~PolSelectRAM ();

      void bypass ();

      void transform_TSPF ();

      void transform_SFPT ();

    protected:
    
    private:

  };
}

#endif
