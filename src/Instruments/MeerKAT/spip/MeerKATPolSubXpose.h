
#ifndef __MeerKATPolSubXpose_h
#define __MeerKATPolSubXpose_h

#include "config.h"

#include "spip/ReadReadWriteBlockClient.h"

namespace spip {

  class MeerKATPolSubXpose : public ReadReadWriteBlockClient {

    public:

      MeerKATPolSubXpose (const char * read1_key_string,
                          const char * read2_key_string,
                          const char * write_key_string,
                          int _subband);

      ~MeerKATPolSubXpose ();

      int64_t open ();

      int64_t io_block (void * read1_buffer, void * read2_buffer, void * write_buffer, uint64_t write_bytes);

      int64_t close ();

    protected:

      int subband;

      int nsubband;

      unsigned nsamp_per_heap;

      unsigned in_stride;

      unsigned out_stride;

      unsigned in1_read_offset;

      unsigned in2_read_offset;

      unsigned in1_write_offset;

      unsigned in2_write_offset;

      unsigned in_length;

      unsigned out_length;

    private:

  };

}

#endif
