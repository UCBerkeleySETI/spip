//-*-C++-*-
/***************************************************************************
 *
 *   Copyright (C) 2018 by Andrew Jameson
 *   Licensed under the Academic Free License version 2.1
 *
 ***************************************************************************/

#ifndef __CUDAtoCUDATransfer_h
#define __CUDAtoCUDATransfer_h

#include "spip/ContainerCUDA.h"
#include "spip/Transformation.h"

#include <cuda_runtime.h>

namespace spip {

  class CUDAtoCUDATransfer: public Transformation <ContainerCUDADevice, ContainerCUDADevice>
  {
    public:
     
      CUDAtoCUDATransfer (cudaStream_t);

      ~CUDAtoCUDATransfer ();

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
