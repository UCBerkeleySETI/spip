//-*-C++-*-
/***************************************************************************
 *
 *   Copyright (C) 2018 by Andrew Jameson
 *   Licensed under the Academic Free License version 2.1
 *
 ***************************************************************************/

#ifndef __AddTimeRAM_h
#define __AddTimeRAM_h

#include "spip/AddTime.h"
#include "spip/ContainerRAM.h"

#include <vector>

namespace spip {

  class AddTimeRAM: public AddTime
  {
    public:
    
      AddTimeRAM ();
      
      ~AddTimeRAM ();

      void configure (Ordering output_order);
 
      void combine_SFPT_to_SFPT ();

      void combine_TFPS_to_TFPS ();

      void combine_TSPF_to_TSPF ();

      void combine_TSPFB_to_TSPFB ();

    protected:
    
    private:

      std::vector<float *> input_buffers; 

  };
}

#endif
