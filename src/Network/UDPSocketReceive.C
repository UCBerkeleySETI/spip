/***************************************************************************
 *
 *   Copyright (C) 2015 Andrew Jameson
 *   Licensed under the Academic Free License version 2.1
 *
 ***************************************************************************/

#include "spip/UDPSocketReceive.h"

#include <arpa/inet.h>

#include <iostream>
#include <cstdlib>
#include <cstring>
#include <cerrno>

#include <stdexcept>

using namespace std;

//! Global flag for receiving function
bool spip::UDPSocketReceive::keep_receiving = true;

spip::UDPSocketReceive::UDPSocketReceive ()
{
  have_packet = 0;
  kernel_bufsz = 131071;      // general default buffer size for linux kernels
  multicast = false;

  // TODO check that this doesn't need to be reassigned
  buf_ptr = buf;
}

spip::UDPSocketReceive::~UDPSocketReceive ()
{
  if (multicast)
    leave_multicast();
  if (buf)
    free (buf);
  buf = 0;
}

// ensure buf_ptr points to the buffer
void spip::UDPSocketReceive::resize (size_t new_bufsz)
{
  spip::Socket::resize (new_bufsz);
  buf_ptr = buf;
}

void spip::UDPSocketReceive::open (string ip_address, int port)
{
#ifdef _DEBUG
  cerr << "spip::UDPSocketReceive::open(" << ip_address << ", " << port << ")" << endl;
#endif

  // open the socket FD
  spip::UDPSocket::open (port);

  if (ip_address.compare("any") == 0)
  {
    udp_sock.sin_addr.s_addr = htonl (INADDR_ANY);
  }
  else
    udp_sock.sin_addr.s_addr = inet_addr (ip_address.c_str());

  // bind socket to file descriptor
  if (bind(fd, (struct sockaddr *)&udp_sock, sizeof(udp_sock)) == -1) 
  {
    throw runtime_error ("could not bind to UDP socket");
  }
  set_nonblock();
}

void spip::UDPSocketReceive::open_multicast (string ip_address, string group, int port)
{
  // open the UDP socket on INADDR_ANY
  open ("any", port);

#ifdef _DEBUG
   cerr << "spip::UDPSocketReceive::open_multicast ip_address=" << ip_address 
        << " group=" << group << " port=" << port << endl;
#endif

  // if we have the XXX.XXX.XXX.XXX+Y notation, then open a sequence of multicast groups
  std::string delimiter = "+";
  size_t pos = group.find(delimiter);

  // if no + notation
  if (pos == std::string::npos)
  {
    num_multicast = 1;
    groups.resize(num_multicast);
    mreqs.resize(num_multicast);
    groups[0] = group;
  }
  else
  {
    // get the XXX.XXX.XXX.XXX
    std::string mcast = group.substr(0, pos);
    // get the +Y
    std::string plus = group.substr(pos + delimiter.length());
    num_multicast = std::stoi (plus) + 1;

#ifdef _DEBUG
   cerr << "spip::UDPSocketReceive::open_multicast num_multicast=" << num_multicast << endl;
#endif
    // build multicast addresses
    groups.resize(num_multicast);
    mreqs.resize(num_multicast);

    std::string period = ".";
    size_t mcast_prefix_pos = mcast.find_last_of(period);
    std::string mcast_prefix = mcast.substr(0, mcast_prefix_pos);
    size_t mcast_suffix = std::stoi(mcast.substr(mcast_prefix_pos+1));

    for (unsigned i=0; i<num_multicast; i++)
    {
      groups[i] = mcast_prefix + "." + std::to_string(mcast_suffix + i);
    }
  }

  for (unsigned i=0; i<num_multicast; i++)
  {
    mreqs[i].imr_multiaddr.s_addr=inet_addr(groups[i].c_str());
    mreqs[i].imr_interface.s_addr=inet_addr(ip_address.c_str());

#ifdef _DEBUG
    cerr << "spip::UDPSocketReceive::open_multicast mreq.imr_multiaddr.s_addr=inet_addr=" << groups[i] << ":" << port << endl;
    cerr << "spip::UDPSocketReceive::open_multicast mreq.imr_interface.s_addr=inet_addr=" << ip_address<< endl;
#endif

    // use setsockopt() to request that the kernel join a multicast group
    if (setsockopt(fd, IPPROTO_IP,IP_ADD_MEMBERSHIP,&(mreqs[i]),sizeof(mreqs[i])) < 0)
    { 
      int perrno = errno;
      cerr << "Error setsockopt: " << strerror(perrno) << endl;
      throw runtime_error ("could not subscribe to multicast address");
    }
  }
  multicast = true;
  set_nonblock();
}

void spip::UDPSocketReceive::leave_multicast ()
{
  for (unsigned i=0; i<num_multicast; i++)
  {
    if (setsockopt(fd, IPPROTO_IP,IP_DROP_MEMBERSHIP,&mreqs[i], sizeof(mreqs[i])) < 0)
    {
      cerr << "could not unsubscribe from multicast address" << endl;
    }
  }
}

size_t spip::UDPSocketReceive::resize_kernel_buffer (size_t pref_size)
{
  int value = 0;
  int len = 0;
  int retval = 0;

  // Attempt to set to the specified value
  value = pref_size;
  len = sizeof(value);
  retval = setsockopt(fd, SOL_SOCKET, SO_RCVBUF, &value, len);
  if (retval != 0)
    throw runtime_error("could not set SO_RCVBUF to default size");

  // now check if it worked
  len = sizeof(value);
  value = 0;
  retval = getsockopt(fd, SOL_SOCKET, SO_RCVBUF, &value, (socklen_t *) &len);
  if (retval != 0)
    throw runtime_error("could not get SO_RCVBUF size");

  // Check the size. n.b. linux actually sets the size to DOUBLE the value
  if (value*2 != pref_size && value/2 != pref_size)
  {
    len = sizeof(value);
    value = 131071;
    retval = setsockopt(fd, SOL_SOCKET, SO_RCVBUF, &value, len);
    if (retval != 0)
      throw runtime_error("could not set SO_RCVBUF size");

    // Now double check that the buffer size is at least correct here
    len = sizeof(value);
    value = 0;
    retval = getsockopt(fd, SOL_SOCKET, SO_RCVBUF, &value, (socklen_t *) &len);
    if (retval != 0)
      throw runtime_error("could not get SO_RCVBUF size");
  }

  return value;
}

size_t spip::UDPSocketReceive::clear_buffered_packets ()
{
  size_t bytes_cleared = 0;
  size_t bytes_read = 0;
  unsigned keep_reading = 1;
  int errsv;

  int was_blocking = get_blocking();

  if (was_blocking)
    set_nonblock();

  while ( keep_reading)
  {
    bytes_read = recvfrom (fd, buf, bufsz, 0, NULL, NULL);
    if (bytes_read == bufsz)
    {
      bytes_cleared += bytes_read;
    }
    else if (bytes_read == -1)
    {
      keep_reading = 0;
      errsv = errno;
      if (errsv != EAGAIN)
        throw runtime_error ("recvfrom failed");
    }
    else
      keep_reading = 0;
  }

  if (was_blocking)
    set_block();

  return bytes_cleared;
}


size_t spip::UDPSocketReceive::recv ()
{
  size_t received = recvfrom (fd, buf, bufsz, 0, NULL, NULL);
  if (received < 0)
  {
    //cerr << "sock_recv recvfrom" << endl;
    return -1;
  }

  return received;
}

size_t spip::UDPSocketReceive::recv_from()
{
  while (!have_packet && keep_receiving)
  {
    pkt_size = (int) recvfrom (fd, buf, bufsz, 0, NULL, NULL);
    if (pkt_size > 32)
    {
      have_packet = true;
    }
    else if (pkt_size == -1)
    {
      nsleeps++;
      if (nsleeps > 1000)
      {
        nsleeps -= 1000;
      }
    }
    else
    {
      cerr << "spip::UDPSocketReceive error expected " << bufsz << " B, "
           << "received " << pkt_size << " B" <<  endl;
      have_packet = true;
      pkt_size = 0;
    }
  }
  return pkt_size;
}

uint64_t spip::UDPSocketReceive::process_sleeps ()
{
  uint64_t accumulated_sleeps = nsleeps;
  nsleeps = 0;
  return accumulated_sleeps;
}


