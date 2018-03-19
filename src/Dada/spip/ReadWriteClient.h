
#ifndef __ReadWriteClient_h
#define __ReadWriteClient_h

#include "config.h"

#include "spip/DadaClient.h"
#include "spip/DataBlockRead.h"
#include "spip/DataBlockWrite.h"

namespace spip {

  class ReadWriteClient : public DadaClient {

    public:

      ReadWriteClient (const char * in_key_string, const char * out_key_string);

      ~ReadWriteClient ();

      // main thread of execution
      int64_t main ();

      // the primary data transfer loop
      int64_t io_loop ();

      // prepare resources necessary for io_data
      void prepare ();

      // method to transfer 1 chunk of data of size optimal_bytes
      virtual int64_t io_data (void * in_data, void * out_data, int64_t bytes) = 0;

    protected:

      // data block from which data is read
      DataBlockRead * read_db;

      // data block to which data is written
      DataBlockWrite * write_db;

    private:

      int64_t bytes_transferred_loop;

      void * read_buffer;

      void * write_buffer;

      size_t buffer_size;

  };

}

#endif
