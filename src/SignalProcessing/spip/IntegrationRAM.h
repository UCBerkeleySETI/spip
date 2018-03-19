//-*-C++-*-
/***************************************************************************
 *
 *   Copyright (C) 2018 by Andrew Jameson
 *   Licensed under the Academic Free License version 2.1
 *
 ***************************************************************************/

#ifndef __IntegrationRAM_h
#define __IntegrationRAM_h

#include "config.h"

#include "spip/Integration.h"
#include "spip/ContainerRAM.h"

namespace spip {

  class IntegrationRAM: public Integration
  {
    public:
    
      IntegrationRAM ();
      
      ~IntegrationRAM ();
 
      void configure (Ordering output_order);

      void prepare ();
      
      void reserve ();
   
      void transform_SFPT_to_SFPT ();

      void transform_TSPF_to_TSPF ();

      void transform_TFPS_to_TFPS ();
      
    protected:
    
    private:

  };
}

#endif
