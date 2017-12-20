
#ifndef __MeerKATPolSubXposeCUDA_h
#define __MeerKATPolSubXposeCUDA_h

#include "config.h"

#include "spip/MeerKATPolSubXpose.h"
#include "spip/CudaClient.h"

#include <cuda_runtime.h>

namespace spip {

  class MeerKATPolSubXposeCUDA : public MeerKATPolSubXpose, public CudaClient {

    public:

      MeerKATPolSubXposeCUDA (const char * read1_key,
                              const char * read2_key,
                              const char * write_key,
                              int _subband, 
                              int _device_id);

      virtual ~MeerKATPolSubXposeCUDA ();

      int64_t open ();

      int64_t io_block (void * read1_buffer, void * read2_buffer, void * write_buffer, uint64_t write_bytes);

      int64_t close ();

    protected:

    private:

      int subband;

  };

}

#endif
