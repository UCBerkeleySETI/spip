
#ifndef __TCPSocket_h
#define __TCPSocket_h

#include "spip/Socket.h"

#include <netinet/in.h>

namespace spip {

  class TCPSocket : public Socket {

    public:

      TCPSocket ();

      ~TCPSocket ();

      // open the socket
      void open (int);

      void close_me ();

      std::string read_bytes (size_t bytes_to_read);

      ssize_t write_bytes (char * buffer, size_t bytes_to_write);

    protected:

      struct sockaddr_in server;

  };

}

#endif
