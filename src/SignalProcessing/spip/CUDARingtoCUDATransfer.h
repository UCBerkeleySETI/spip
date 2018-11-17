//-*-C++-*-
/***************************************************************************
 *
 *   Copyright (C) 2018 by Andrew Jameson
 *   Licensed under the Academic Free License version 2.1
 *
 ***************************************************************************/

#ifndef __CUDARingtoCUDATransfer_h
#define __CUDARingtoCUDATransfer_h

#include "spip/ContainerRingReadCUDA.h"
#include "spip/ContainerCUDA.h"
#include "spip/Transformation.h"

#include <cuda_runtime.h>

namespace spip {

  class CUDARingtoCUDATransfer: public Transformation <ContainerRingReadCUDA, ContainerCUDADevice>
  {
    public:
     
      CUDARingtoCUDATransfer (cudaStream_t);

      ~CUDARingtoCUDATransfer ();

      void set_output_reblock (unsigned);

      void configure (Ordering output_order);

      void prepare ();

      void prepare_output ();

      void reserve ();

      void transformation ();

    protected:

    private:

      cudaStream_t stream;

      uint64_t ndat;

      unsigned iblock_out;

      unsigned nblock_out;

  };

}

#endif
