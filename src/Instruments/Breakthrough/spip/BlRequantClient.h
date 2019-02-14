
#ifndef __BlRequantClient_h
#define __BlRequantClient_h

#include "config.h"

#include "spip/ReadWriteBlockClient.h"

namespace spip {

  class BlRequantClient : public ReadWriteBlockClient {

    public:

      BlRequantClient (const char * read_key_string,
                          const char * write_key_string);

      ~BlRequantClient ();

      int64_t open ();

      int64_t io_block (void * read_buffer, void * write_buffer, uint64_t read_bytes);

      int64_t close ();

    protected:

      float scale_factor;

      unsigned sample_bytes;

    private:

      inline int16_t  convert_twos (int16_t in)  { if (in == 0) return 0; else return in ^ 0x8000; };

      inline int16_t  convert_twos_fast (int16_t in)  { return in ^ 0x8000; };

      inline int32_t convert_twos_pair (int32_t in) { if (in == 0) return 0; else return  in ^ 0x80008000; };

      inline int32_t convert_twos_pair_fast (int32_t in) { return  in ^ 0x80008000; };
  };

}

#endif
