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
  vma_pkts = NULL;
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
    while (!have_packet && keep_receiving)
    {
      if (vma_pkts)
      {
        vma_api->free_packets(fd, vma_pkts->pkts, vma_pkts->n_packet_num);
        vma_pkts = NULL;
      }

      int flags = 0;
      int size = vma_api->recvfrom_zcopy(fd, buf, bufsz, &flags, NULL, NULL);

      // largest useful sized packet
      if (size > 0)
      {
        if ((flags & MSG_VMA_ZCOPY) == MSG_VMA_ZCOPY)
        {
          vma_pkts = (vma_packets_t*) buf;
          if (vma_pkts->n_packet_num == 1)
          {
            if (vma_pkts->pkts[0].sz_iov == 1)
            {
              // AJ to check this!
              pkt_size = vma_pkts->pkts[0].iov[0].iov_len - 2;
              buf_ptr = (char *) vma_pkts->pkts[0].iov[0].iov_base;
              have_packet = true;
            }
            else
            {
              cerr << "spip::UDPSocketReceiveVMA::recv_from pkts->pkts[0].sz_iov=" << vma_pkts->pkts[0].sz_iov << endl;
            }
          }
          else
          {
            cerr << "spip::UDPSocketReceiveVMA::recv_from pkts->n_packet_num=" << vma_pkts->n_packet_num << endl;
          }
        }
        else
        {
          pkt_size = size;
          buf_ptr = buf;
          have_packet = true;
        }
      }
      else if (pkt_size == -1)
      {
        throw runtime_error ("UDPSocketReceiveVMA requires blocking sockets");
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


