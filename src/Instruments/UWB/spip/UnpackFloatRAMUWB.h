//-*-C++-*-
/***************************************************************************
 *
 *   Copyright (C) 2018 by Andrew Jameson
 *   Licensed under the Academic Free License version 2.1
 *
 ***************************************************************************/

#ifndef __UnpackFloatRAMUWB_h
#define __UnpackFloatRAMUWB_h

#include "config.h"

#include "spip/UnpackFloatRAM.h"

#include <iostream>
#include <climits>

namespace spip {

  class UnpackFloatRAMUWB: public UnpackFloatRAM
  {
    public:
    
      UnpackFloatRAMUWB ();
      
      ~UnpackFloatRAMUWB ();
 
      void prepare ();
      
      void transform_custom_to_SFPT ();
      
    protected:
    
    private:

      inline int16_t  convert_twos (int16_t in)  { if (in == 0) return 0; else return in ^ 0x8000; };

      float re_scale;
      float im_scale;

  };
}

#endif
