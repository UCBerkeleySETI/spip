//-*-C++-*-
/***************************************************************************
 *
 *   Copyright (C) 2017 by Andrew Jameson
 *   Licensed under the Academic Free License version 2.1
 *
 ***************************************************************************/

#ifndef __RAMtoCUDATransfer_h
#define __RAMtoCUDATransfer_h

#include "spip/ContainerCUDA.h"
#include "spip/Transformation.h"

#include <cuda_runtime.h>

namespace spip {

  class RAMtoCUDATransfer: public Transformation <Container, ContainerCUDADevice>
  {
    public:
     
      RAMtoCUDATransfer (cudaStream_t);

      ~RAMtoCUDATransfer ();

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

  };

}

#endif
