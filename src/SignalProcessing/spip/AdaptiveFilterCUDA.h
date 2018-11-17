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
#include "spip/ContainerCUDA.h"
#include "spip/ContainerCUDAFileWrite.h"
#include <cuda_runtime.h>

namespace spip {

  class AdaptiveFilterCUDA: public AdaptiveFilter
  {
    public:
    
      AdaptiveFilterCUDA (cudaStream_t, const std::string&);
      
      ~AdaptiveFilterCUDA ();

      void configure (Ordering output_order);

      void transform_TSPF ();

      void transform_SFPT ();

      void write_gains ();

      void write_dirty ();

      void write_cleaned ();

    protected:
    
    private:

      cudaStream_t stream;

      bool processed_first_block;

      ContainerCUDAFileWrite * gains_file_write;

      ContainerCUDAFileWrite * dirty_file_write;

      ContainerCUDAFileWrite * cleaned_file_write;
  };
}

#endif
