//-*-C++-*-
/***************************************************************************
 *
 *   Copyright (C) 2018 by Andrew Jameson
 *   Licensed under the Academic Free License version 2.1
 *
 ***************************************************************************/

#ifndef __RINGtoCUDATransfer_h
#define __RINGtoCUDATransfer_h

#include "spip/ContainerRingRead.h"
#include "spip/ContainerCUDA.h"
#include "spip/Transformation.h"

#include <cuda_runtime.h>

namespace spip {

  class RINGtoCUDATransfer: public Transformation <ContainerRingRead, ContainerCUDADevice>
  {
    public:
     
      RINGtoCUDATransfer (cudaStream_t);

      ~RINGtoCUDATransfer ();

      void set_output_reblock (unsigned);

      void configure (Ordering output_order);

      void prepare ();

      void prepare_output ();

      void reserve ();

      //! Perform a forward FFT on input block
      void transformation ();

    protected:

    private:

      cudaStream_t stream;

      uint64_t ndat;

      unsigned iblock_out;

      unsigned nblock_out;

      cudaMemcpyKind kind;

  };

}

#endif
