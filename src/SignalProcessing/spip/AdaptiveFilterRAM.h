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
#include "spip/ContainerRAMFileWrite.h"

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

      void write_dirty ();

      void write_cleaned ();

    protected:
    
    private:

      float * ast_buffer;

      float * ref_buffer;

      size_t buffer_size;

      ContainerRAMFileWrite * gains_file_write;

      ContainerRAMFileWrite * dirty_file_write;

      ContainerRAMFileWrite * cleaned_file_write;

      float previous_factor;

      bool processed_first_block;

  };
}

#endif
