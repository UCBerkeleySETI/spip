
#ifndef __UDPSocketSend_h
#define __UDPSocketSend_h

#include "spip/UDPSocket.h"

#include <netinet/in.h>

namespace spip {

  class UDPSocketSend : public UDPSocket {

    public:

      UDPSocketSend ();

      virtual ~UDPSocketSend ();

      // open the socket
      void open (std::string, int, std::string);

      // open the socket and bind to a multicast group
      void open_multicast (std::string, int, std::string);

      // send the contents of buf (bufsz bytes)
      inline size_t send () { return sendto(fd, buf, bufsz, 0, sock_addr, sock_size); };

      size_t send (size_t nbytes);

    private:

      struct sockaddr * sock_addr;

      size_t sock_size;

      std::string group;

  };

}

#endif
