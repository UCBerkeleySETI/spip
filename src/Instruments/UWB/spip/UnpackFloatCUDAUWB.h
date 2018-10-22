//-*-C++-*-
/***************************************************************************
 *
 *   Copyright (C) 2018 by Andrew Jameson
 *   Licensed under the Academic Free License version 2.1
 *
 ***************************************************************************/

#ifndef __UnpackFloatCUDAUWB_h
#define __UnpackFloatCUDAUWB_h

#include "config.h"

#include "spip/UnpackFloatCUDA.h"

#include <iostream>
#include <climits>

namespace spip {

  class UnpackFloatCUDAUWB: public UnpackFloatCUDA
  {
    public:
    
      UnpackFloatCUDAUWB ();
      
      ~UnpackFloatCUDAUWB ();
 
      void prepare ();
      
      void transform_custom_to_SFPT ();
      
    protected:
    
    private:

  };
}

#endif
