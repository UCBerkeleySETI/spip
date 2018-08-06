//-*-C++-*-
/***************************************************************************
 *
 *   Copyright (C) 2018 by Andrew Jameson
 *   Licensed under the Academic Free License version 2.1
 *
 ***************************************************************************/

#ifndef __IntegrationBinnedRAM_h
#define __IntegrationBinnedRAM_h

#include "config.h"

#include "spip/IntegrationBinned.h"
#include "spip/ContainerRAM.h"

namespace spip {

  class IntegrationBinnedRAM: public IntegrationBinned
  {
    public:
    
      IntegrationBinnedRAM ();
      
      ~IntegrationBinnedRAM ();
 
      void configure (Ordering output_order);

      void transform_TSPF_to_TSPFB ();

    protected:
    
    private:

  };
}

#endif
