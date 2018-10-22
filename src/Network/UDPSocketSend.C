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

//#define _DEBUG

using namespace std;

spip::UDPSocketSend::UDPSocketSend ()
{
  sock_addr = (struct sockaddr *) &udp_sock;
  sock_size = sizeof(struct sockaddr);
}

spip::UDPSocketSend::~UDPSocketSend ()
{
}

void spip::UDPSocketSend::open (string ip_address, int port, string local_ip_address)
{
#ifdef _DEBUG
  cerr << "spip::UDPSocketSend::open sending to " << ip_address << ":" << port << " from " << local_ip_address << endl;
#endif
  // open the socket FD
  spip::UDPSocket::open (port);

  // transmitting sockets must have an IP specified
  struct in_addr *addr;
  addr = atoaddr (ip_address.c_str());
  udp_sock.sin_addr.s_addr = addr->s_addr;

  if (local_ip_address.compare("any") == 0)
  {
    other_udp_sock.sin_addr.s_addr = htonl (INADDR_ANY);
  }
  else
    other_udp_sock.sin_addr.s_addr = inet_addr (local_ip_address.c_str());

  // bind socket to file descriptor
  if (bind(fd, (struct sockaddr *)&other_udp_sock, sizeof(other_udp_sock)) == -1)
  {
    throw runtime_error ("could not bind to UDP socket");
  }

}

void spip::UDPSocketSend::open_multicast (string group_address, int port, string local_ip_address)
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
#ifdef _DEBUG
    cerr << "spip::UDPSocketSend::open_multicast only using first multicast group" << endl;
#endif

    // get the XXX.XXX.XXX.XXX
    std::string mcast = group_address.substr(0, pos);
    // get the +Y
    std::string plus = group_address.substr(pos + delimiter.length());

    std::string period = ".";
    size_t mcast_prefix_pos = mcast.find_last_of(period);
    std::string mcast_prefix = mcast.substr(0, mcast_prefix_pos);
    size_t mcast_suffix = std::stoi(mcast.substr(mcast_prefix_pos+1));

#ifdef _DEBUG
    cerr << "mcast_prefix=" << mcast_prefix << " mcast_suffix=" << mcast_suffix << endl;
#endif

    group = mcast_prefix + "." + std::to_string(mcast_suffix);
  }

#ifdef _DEBUG
  cerr << "spip::UDPSocketSend::open_multicast open(" << group << ", " << port << ")" << endl;
#endif
  open (group, port, local_ip_address);
}


size_t spip::UDPSocketSend::send (size_t nbytes)
{
  if (nbytes > bufsz)
    throw runtime_error ("cannot send more bytes than socket size");
  return sendto(fd, buf, nbytes, 0, sock_addr, sock_size); 
}

