
#ifndef __UDPSocketReceive_h
#define __UDPSocketReceive_h

#include "config.h"
#include "spip/UDPSocket.h"

#include <vector>
#include <netinet/in.h>

namespace spip {

  class UDPSocketReceive : public UDPSocket {

    public:

      //! Global flag to cease receiving
      static bool keep_receiving;

      UDPSocketReceive ();

      ~UDPSocketReceive ();

      // resize the socket
      void resize (size_t new_bufsz);

      // open the socket
      virtual void open (std::string, int);

      void bind (int);

      // open the socket and bind to a multicast group
      virtual void open_multicast (std::string, std::string, int port);

      // leave a multicast group on socket
      void leave_multicast ();

      size_t resize_kernel_buffer (size_t);

      size_t clear_buffered_packets ();

      size_t recv ();

      inline void consume_packet() { have_packet = false; };

      inline bool still_receiving() { return keep_receiving; } ;

      uint64_t process_sleeps ();

      // pointer to the socket buffer, may be reassigned
      char * buf_ptr;

      // receive a packet from the UDP socket
      ssize_t recv_from ();

    protected:

      // flag for whether bufsz contains a packet
      bool have_packet;

      ssize_t pkt_size;

    private:

      // size of the kernel socket buffer;
      size_t kernel_bufsz;

      // if the receiving socket is multicast
      bool multicast;

      size_t num_multicast;

      std::vector<std::string> groups;

      std::vector<struct ip_mreq> mreqs;

      uint64_t nsleeps;

  };

}

#endif
