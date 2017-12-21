
#ifndef __ReadWriteBlockClient_h
#define __ReadWriteBlockClient_h

#include "config.h"

#include "spip/DadaClient.h"
#include "spip/DataBlockRead.h"
#include "spip/DataBlockWrite.h"

namespace spip {

  class ReadWriteBlockClient : public DadaClient {

    public:

      ReadWriteBlockClient (const char * read_key_string,
                            const char * write_key_string);

      ~ReadWriteBlockClient ();

      // main thread of execution
      int64_t main ();

      // perform tests and configuration
      void configure ();

      // 
      int64_t open () = 0;

      // the primary data transfer loop
      int64_t io_loop ();

      int64_t close () = 0;

      // prepare resources necessary for io_data
      void prepare ();

      // method to transfer input data to output data 
      virtual int64_t io_block (void * in_data, void * out_data, uint64_t bytes) = 0;

    protected:

      // data block from which data is read
      DataBlockRead * read_db;

      // data block to which data is written
      DataBlockWrite * write_db;

      // header read from read_db
      AsciiHeader read_header;

      // header written to write_db
      AsciiHeader write_header;

    private:

      int64_t bytes_transferred_loop;

      void * read_buffer;

      void * write_buffer;

      size_t read_buffer_size;

      size_t write_buffer_size;

      uint64_t read_block_size;

      uint64_t write_block_size;

  };

}

#endif
