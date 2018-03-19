
#ifndef __WriteBlockClient_h
#define __WriteBlockClient_h

#include "config.h"

#include "spip/DadaClient.h"
#include "spip/DataBlockWrite.h"

namespace spip {

  class WriteBlockClient : public DadaClient {

    public:

      WriteBlockClient (const char * key_string);

      ~WriteBlockClient ();

      int64_t main ();

      int64_t io_loop ();

      virtual int64_t io_block (void * data, int64_t bytes) = 0;

    protected:

      // data block that is written to
      DataBlockWrite * db;

    private:

      int64_t bytes_written_loop;

  };

}

#endif
