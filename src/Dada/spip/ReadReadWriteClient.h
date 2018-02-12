
#ifndef __ReadReadWriteClient_h
#define __ReadReadWriteClient_h

#include "config.h"

#include "spip/DadaClient.h"
#include "spip/DataBlockRead.h"
#include "spip/DataBlockWrite.h"

namespace spip {

  class ReadReadWriteClient : public DadaClient {

    public:

      ReadReadWriteClient (const char * in1_key_string, const char * in2_key_string, const char * out_key_string);

      ~ReadReadWriteClient ();

      // main thread of execution
      int64_t main ();

      // the primary data transfer loop
      int64_t io_loop ();

      // prepare resources necessary for io_data
      void prepare ();

      // method to transfer 2 sets in input data, writing bytes output data 
      virtual int64_t io_data (void * in1_data, void * in2_data, void * out_data, int64_t bytes) = 0;

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

  };

}

#endif
