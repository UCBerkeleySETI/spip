/***************************************************************************
 *
 *   Copyright (C) 2015 Andrew Jameson
 *   Licensed under the Academic Free License version 2.1
 *
 ***************************************************************************/

#include "spip/UDPSocketSend.h"

#include <arpa/inet.h>
#include <netdb.h>

#include <cstdlib>
#include <cstring>
#include <cerrno>
#include <iostream>
#include <stdexcept>

using namespace std;

spip::UDPSocketSend::UDPSocketSend ()
{
  sock_addr = (struct sockaddr *) &udp_sock;
  sock_size = sizeof(struct sockaddr);
}

spip::UDPSocketSend::~UDPSocketSend ()
{
}

void spip::UDPSocketSend::open (string ip_address, int port)
{
  // open the socket FD
  spip::UDPSocket::open (port);

  // transmitting sockets must have an IP specified
  struct in_addr *addr;
  addr = atoaddr (ip_address.c_str());
  udp_sock.sin_addr.s_addr = addr->s_addr;
}

void spip::UDPSocketSend::open_multicast (string group_address, int port)
{
  // if we have the XXX.XXX.XXX.XXX+Y notation, then open a sequence of multicast groups
  std::string delimiter = "+";
  size_t pos = group_address.find(delimiter);

  // if no + notation
  if (pos == std::string::npos)
  {
    group = group_address;
  }
  else
  {
    cerr << "spip::UDPSocketSend::open_multicast only using first multicast group" << endl;

    // get the XXX.XXX.XXX.XXX
    std::string mcast = group_address.substr(0, pos);
    // get the +Y
    std::string plus = group_address.substr(pos + delimiter.length());

    std::string period = ".";
    size_t mcast_prefix_pos = mcast.find_last_of(period);
    std::string mcast_prefix = mcast.substr(0, mcast_prefix_pos);
    size_t mcast_suffix = std::stoi(mcast.substr(mcast_prefix_pos+1));

    cerr << "mcast_prefix=" << mcast_prefix << " mcast_suffix=" << mcast_suffix << endl;

    group = mcast_prefix + "." + std::to_string(mcast_suffix);
  }

  cerr << "spip::UDPSocketSend::open_multicast open(" << group << ", " << port << ")" << endl;
  open (group, port);
}


size_t spip::UDPSocketSend::send (size_t nbytes)
{
  if (nbytes > bufsz)
    throw runtime_error ("cannot send more bytes than socket size");
  return sendto(fd, buf, nbytes, 0, sock_addr, sock_size); 
}

