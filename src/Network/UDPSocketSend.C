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

void spip::UDPSocketSend::open (string ip_address, int port, string local_ip_address)
{
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

size_t spip::UDPSocketSend::send (size_t nbytes)
{
  if (nbytes > bufsz)
    throw runtime_error ("cannot send more bytes than socket size");
  return sendto(fd, buf, nbytes, 0, sock_addr, sock_size); 
}

