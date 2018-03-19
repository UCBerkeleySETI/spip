//-*-C++-*-
/***************************************************************************
 *
 *   Copyright (C) 2017 by Andrew Jameson
 *   Licensed under the Academic Free License version 2.1
 *
 ***************************************************************************/

#ifndef __CUDAtoRAMTransfer_h
#define __CUDAtoRAMTransfer_h

#include "spip/ContainerRAM.h"
#include "spip/ContainerCUDA.h"
#include "spip/Transformation.h"

#include <cuda_runtime.h>

namespace spip {

  class CUDAtoRAMTransfer: public Transformation <ContainerCUDADevice, Container>
  {
    public:
     
      CUDAtoRAMTransfer (cudaStream_t);

      ~CUDAtoRAMTransfer ();

      void configure (Ordering output_order);

      void prepare ();

      void prepare_output ();

      void reserve ();

      void transformation ();

    protected:

    private:

      cudaStream_t stream;

      uint64_t ndat;

  };

}

#endif
