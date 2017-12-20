
#ifndef __ReadClient_h
#define __ReadClient_h

#include "config.h"

#include "spip/DadaClient.h"
#include "spip/DataBlockRead.h"

namespace spip {

  class ReadClient : public DadaClient {

    public:

      ReadClient (const char * key_string);

      ~ReadClient ();

      int64_t main ();

      int64_t io_loop ();

      virtual void io (void * data, int64_t bytes) = 0;

    protected:

      // data block that is written to
      DataBlockRead * db;

    private:

      int64_t bytes_read_loop;

  };

}

#endif
