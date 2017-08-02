
#ifndef __UDPSocketReceiveVMA_h
#define __UDPSocketReceiveVMA_h

#include "spip/UDPSocketReceive.h"

#include <vector>
#include <netinet/in.h>
#include <mellanox/vma_extra.h>

namespace spip {

  class UDPSocketReceiveVMA : public UDPSocketReceive {

    public:

      UDPSocketReceiveVMA ();

      ~UDPSocketReceiveVMA ();

      // open the socket
      void open (std::string, int);

      // open the socket and bind to a multicast group
      void open_multicast (std::string, std::string, int port);

      // VMA specific recv_from 
      size_t recv_from ();

    private:

      struct vma_api_t * vma_api;

      struct vma_packets_t* pkt = NULL;

  };

}

#endif
