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
#include "spip/ContainerFileWrite.h"

#include <string.h>

namespace spip {

  class AdaptiveFilterRAM: public AdaptiveFilter
  {
    public:
    
      AdaptiveFilterRAM (std::string);
      
      ~AdaptiveFilterRAM ();

      void set_input_ref (Container *);
      
      void configure (Ordering order);

      void prepare ();
      
      void reserve ();
      
      void transform_TSPF ();

      void transform_SFPT ();

      void write_gains ();

    protected:
    
    private:

      float * ast_buffer;

      float * ref_buffer;

      size_t buffer_size;

      ContainerFileWrite * gains_file_write;
  };
}

#endif
