
#ifndef __TCPSocketServer_h
#define __TCPSocketServer_h

#include "spip/TCPSocket.h"

#include <netinet/in.h>

namespace spip {

  class TCPSocketServer : public TCPSocket {

    public:

      TCPSocketServer ();

      ~TCPSocketServer ();

      // open the listening socket
      void open (std::string, int, int);

      // close the listening socket
      void close_me ();

      // enter blocking accept call
      int accept_client ();

      // try to accept a connection within a timeout
      int accept_client (int timeout);

      // close a client socket connection
      void close_client ();

      // read an string from the client socket
      std::string read_client (size_t bytes_to_recv);

      ssize_t write_client (char * buffer, size_t bytes);

      // configure verbosity
      void set_verbosity (int level);

      // indicate sockets are closing
      void set_ending () { ending = true; } ;

    protected:

      int verbose;

    private:

      int client_fd;

      bool ending;

  };

}

#endif
