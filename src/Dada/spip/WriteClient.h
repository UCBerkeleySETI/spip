
#ifndef __WriteClient_h
#define __WriteClient_h

#include "config.h"

#include "spip/DadaClient.h"
#include "spip/DataBlockWrite.h"

namespace spip {

  class WriteClient : public DadaClient {

    public:

      WriteClient (const char * key_string);

      ~WriteClient ();

      int64_t main ();

      int64_t io_loop ();

      virtual int64_t io (void * data, int64_t bytes) = 0;

    protected:

      // data block that is written to
      DataBlockWrite * db;

    private:

      int64_t bytes_written_loop;

  };

}

#endif
