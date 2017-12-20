
#ifndef __DadaClient_h
#define __DadaClient_h

#include "config.h"

#include "spip/AsciiHeader.h"
#include "spip/Error.h"

#include <iostream>
#include <cstdlib>
#include <pthread.h>

namespace spip {

  class DadaClient {

    public:

      DadaClient ();

      ~DadaClient ();

      // open data transfer to the target
      virtual int64_t open () = 0;

      // perform buffered on the data block
      // virtual int64_t io_data (void * data, uint64_t data_size) = 0;

      // perform direct I/O transfer 
      // virtual int64_t io_block (void * data, uint64_t data_size) = 0;

      // perform direct I/O transfer with device memory
      // virtual int64_t io_block_cuda (void * data, uint64_t data_size) = 0;

      // close data transfer to the target
      virtual int64_t close () = 0;

      // main processing loop
      virtual int64_t main () = 0;

    protected:

      // verbosity flag
      bool verbose;

      // header to be transferred to the target
      AsciiHeader header;

      // total number of bytes to transfer to/from the target
      int64_t transfer_bytes;

      // the optimal number of bytes to transfer at a time to/from the target
      int64_t optimal_bytes;

      // a temporary buffer to transfer data between target and DB
      char * buffer;

      // size of the temporary buffer
      size_t buffer_size;

  };

}

#endif
