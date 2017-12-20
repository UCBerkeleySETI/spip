#ifndef __CudaClient_h
#define __CudaClient_h

#include "config.h"
#include "spip/Error.h"

#include <cuda_runtime.h>

namespace spip {

  class CudaClient {

    public:

      CudaClient (int _device_id);

      ~CudaClient ();

    protected:

      int device_id;

      int device_count;

      cudaStream_t stream;

      cudaDeviceProp device_prop;

  };

}

#endif
