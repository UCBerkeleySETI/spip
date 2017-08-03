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
  pkts = NULL;
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
  set_block ();
}

void spip::UDPSocketReceiveVMA::open_multicast (string ip_address, string group, int port)
{
#ifdef _DEBUG
  cerr << "spip::UDPSocketReceiveVMA::open_multicast(" << ip_address << "," << group << ", " << port << ")" << endl;
#endif

  // open the socket FD
  spip::UDPSocketReceive::open_multicast (ip_address, group, port);

  // VMA sockets are blocking
  set_block ();
}


size_t spip::UDPSocketReceiveVMA::recv_from()
{
  if (vma_api)
  {
    if (pkts)
    {
      vma_api->free_packets(fd, pkts->pkts, pkts->n_packet_num);
      pkts = NULL;
    }
    while (!have_packet && keep_receiving)
    {
      int flags = 0;
      pkt_size = (int) vma_api->recvfrom_zcopy(fd, buf, bufsz, &flags, NULL, NULL);
      
      // largest useful sized packet
      if (pkt_size > 32)
      {
        if (flags == MSG_VMA_ZCOPY)
        {
          pkts = (vma_packets_t*) buf;
          if (pkts->n_packet_num == 1)
          {
            buf_ptr = (char *) pkts->pkts[0].iov[0].iov_base;
            have_packet = true;
          }
        }
        else
          buf_ptr = buf;
      }
      else if (pkt_size == -1)
      {
        throw runtime_error ("UDPSocketReceiveVMA requires non blocking sockets");
      }
      else
      {
        cerr << "spip::UDPReceiveSocketVMA error expected " << bufsz
             << " B, received " << pkt_size << " B" <<  endl;
        keep_receiving = false;
        pkt_size = 0;
      }
    }
    return pkt_size;
  }
  else
    return spip::UDPSocketReceive::recv_from();
}


