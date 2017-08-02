/***************************************************************************
 *
 *   Copyright (C) 2015 Andrew Jameson
 *   Licensed under the Academic Free License version 2.1
 *
 ***************************************************************************/

#include "spip/UDPSocketReceiveVMA.h"

#include <arpa/inet.h>

#include <iostream>
#include <cstdlib>
#include <cstring>
#include <cerrno>

#include <stdexcept>

using namespace std;

spip::UDPSocketReceiveVMA::UDPSocketReceiveVMA ()
{
  vma_api = vma_get_api();
  pkt = NULL;
  if (!vma_api)
  {
    cerr << "WARNING: VMA support compiled, but VMA not available" << endl;
  }
}

spip::UDPSocketReceiveVMA::~UDPSocketReceiveVMA ()
{
}

void spip::UDPSocketReceiveVMA::open (string ip_address, int port)
{
#ifdef _DEBUG
  cerr << "spip::UDPSocketReceiveVMA::open(" << ip_address << ", " << port << ")" << endl;
#endif

  // open the socket FD
  spip::UDPSocketReceive::open (ip_address, port);

  // VMA sockets are non blocking
  set_nonblock ();
}

void spip::UDPSocketReceive::open_multicast (string ip_address, string group, int port)
{
#ifdef _DEBUG
  cerr << "spip::UDPSocketReceiveVMA::open_multicast(" << ip_address << "," << group << ", " << port << ")" << endl;
#endif

  // open the socket FD
  spip::UDPSocketReceive::open_multicast (ip_address, group, port);

  // VMA sockets are non blocking
  set_nonblock ();
}


size_t spip::UDPSocketReceiveVMA::recv_from()
{
  if (vma_api)
  {
    int got = 0;
    if (pkt)
    {
      vma_api->free_packets(fd, pkt->pkts, pkt->n_packet_num);
      pkt = NULL;
    }
    while (!have_packet && keep_receiving)
    {
      int flags = 0;
      //got = (int) vma_api->recvfrom_zcopy(fd, buf, bufsz, &flags, addr, &addr_size);
      got = (int) vma_api->recvfrom_zcopy(fd, buf, bufsz, &flags, NULL, NULL);
      if (got  > 32)
      {
        if (flags == MSG_VMA_ZCOPY)
        {
          pkt = (vma_packets_t*) buf;
          buf_ptr = (char *) pkt->pkts[0].iov[0].iov_base;
        }
        have_packet = true;
      }
      else if (got == -1)
      {
        throw runtime_error ("UDPSocketReceiveVMA requires non blocking sockets");
      }
      else
      {
        cerr << "spip::UDPReceiveSocketVMA error expected " << bufsz
             << " B, received " << got << " B" <<  endl;
        keep_receiving = false;
        got = 0;
      }
    }
    return got;
  }
  else
    return spip::UDPSocketReceive::recv_from();
}


