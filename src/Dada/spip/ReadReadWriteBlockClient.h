
#ifndef __ReadReadWriteBlockClient_h
#define __ReadReadWriteBlockClient_h

#include "config.h"

#include "spip/DadaClient.h"
#include "spip/DataBlockRead.h"
#include "spip/DataBlockWrite.h"

namespace spip {

  class ReadReadWriteBlockClient : public DadaClient {

    public:

      ReadReadWriteBlockClient (const char * read1_key_string,
                                const char * read2_key_string,
                                const char * write_key_string);

      ~ReadReadWriteBlockClient ();

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

      // method to transfer 2 sets in input data, writing bytes output data 
      virtual int64_t io_block (void * in1_data, void * in2_data, void * out_data, uint64_t bytes) = 0;

    protected:

      // data block from which data is read
      DataBlockRead * read1_db;

      // data block from which data is read
      DataBlockRead * read2_db;

      // data block to which data is written
      DataBlockWrite * write_db;

      // header read from read1_db
      AsciiHeader read1_header;

      // header read from read2_db
      AsciiHeader read2_header;

      // header written to write_db
      AsciiHeader write_header;

    private:

      int64_t bytes_transferred_loop;

      void * read1_buffer;

      void * read2_buffer;

      void * write_buffer;

      size_t read_buffer_size;

      size_t write_buffer_size;

      uint64_t read1_block_size;

      uint64_t read2_block_size;

      uint64_t write_block_size;

  };

}

#endif
