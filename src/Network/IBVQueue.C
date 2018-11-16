/***************************************************************************
 *
 *   Copyright (C) 2015 Andrew Jameson
 *   Licensed under the Academic Free License version 2.1
 *
 ***************************************************************************/

#include "spip/IBVQueue.h"

#include <fcntl.h>
#include <sys/types.h>
#include <sys/socket.h>
#include <unistd.h>
#include <arpa/inet.h>
#include <netdb.h>

#include <stdexcept>
#include <cstdlib>
#include <iostream>

#include <spead2/common_raw_packet.h>

//#define _DEBUG
//#define _TRACE

using namespace std;
using namespace spead2;

//! Global flag for receiving function
bool spip::IBVQueue::keep_receiving = true;

spip::IBVQueue::IBVQueue (boost::asio::io_service& io_service) : join_socket(io_service, boost::asio::ip::udp::v4())
{
  buffer_size = 65536;
  packet_size = 9000;
  n_slots = 0;

  islot = -1;
  ipacket = 0;
  npackets = 0;
}

spip::IBVQueue::~IBVQueue ()
{
  join_socket.close();

  // drain the irecv queue
  while (ipacket < npackets)
  {
    open_packet();
    close_packet();
  }

  // sleep so that we can unsubscrube from the switch
  sleep(1);
}


void spip::IBVQueue::build (const boost::asio::ip::address_v4 interface_address)
{
#ifdef _DEBUG
  cerr << "spip::IBVQueue::build buffer_size=" << buffer_size << " packet_size=" << packet_size << endl;
#endif
  if (n_slots != 0)
    throw runtime_error ("Queue must be configured prior to building");

  n_slots = int(buffer_size / max_raw_size);

#ifdef _DEBUG
  cerr << "spip::IBVQueue::build buffer_size=" << buffer_size << " packet_size=" << packet_size << " n_slots=" << n_slots << endl;
#endif

#ifdef _DEBUG
  cerr << "spip::IBVQueue::build binding to " << interface_address << endl;
#endif
  cm_id = rdma_cm_id_t (event_channel, nullptr, RDMA_PS_UDP);
  cm_id.bind_addr(interface_address);

  // assume no completion channel
#ifdef _DEBUG
  cerr << "spip::IBVQueue::build creating completion channels" << endl;
#endif
  cq = ibv_cq_t(cm_id, n_slots, nullptr);

  // create the protection domain
#ifdef _DEBUG
  cerr << "spip::IBVQueue::build creating protection domain" << endl;
#endif
  pd = ibv_pd_t(cm_id);
  
  // create the queue pair
#ifdef _DEBUG
  cerr << "spip::IBVQueue::build creating QP" << endl;
#endif
  qp = create_qp (pd, cq, n_slots);

#ifdef _DEBUG
  cerr << "spip::IBVQueue::build qp.modify (IBV_QPS_INIT, cm_id->port_num);" << endl;
#endif
  qp.modify (IBV_QPS_INIT, cm_id->port_num);
}

void spip::IBVQueue::configure (size_t _npackets, size_t _packet_size, size_t _header_size)
{
#ifdef _DEBUG
  cerr << "spip::IBVQueue::configure npackets=" << _npackets
       << " packet_size=" << _packet_size << " header_size=" << _header_size 
       << endl;
#endif
  if (n_slots != 0)
    throw runtime_error ("Queue cannot be configured after building"); 
  packet_size = _packet_size;
  header_size = _header_size;

  size_t ethernet_size = 6 + 6 + 4;
  size_t ipv4_size = 20;
  size_t udp_size = 8;
  size_t vlan_size = 4;
  size_t packet_headers = ethernet_size + ipv4_size + udp_size + vlan_size;

  max_raw_size = packet_headers + header_size + packet_size;
  buffer_size = _npackets * max_raw_size;
#ifdef _DEBUG
  cerr << "spip::IBVQueue::configure slot_size=" << max_raw_size << endl;
  cerr << "spip::IBVQueue::configure buffer_size=" << buffer_size << endl;
#endif
}

void spip::IBVQueue::allocate ()
{
#ifdef _DEBUG
  cerr << "spip::IBVQueue::allocate buffer_size=" << buffer_size<< endl;
#endif
  std::shared_ptr<mmap_allocator> allocator = std::make_shared<mmap_allocator>(0, true);
  buffer = allocator->allocate(buffer_size, nullptr);
  mr = ibv_mr_t(pd, buffer.get(), buffer_size, IBV_ACCESS_LOCAL_WRITE);
#ifdef _DEBUG
  cerr << "spip::IBVQueue::allocate n_slots=" << n_slots<< endl;
#endif
  slots.reset(new slot[n_slots]);
  wc.reset(new ibv_wc[n_slots]);
  for (std::size_t i = 0; i < n_slots; i++)
  {
    std::memset(&slots[i], 0, sizeof(slots[i]));
    //slots[i].wr.next = (i + 1 < n_slots) ? &slots[i+1].wr : nullptr;
    slots[i].sge.addr = (uintptr_t) &buffer[i * max_raw_size];
    slots[i].sge.length = max_raw_size;
    slots[i].sge.lkey = mr->lkey;
    slots[i].wr.sg_list = &slots[i].sge;
    slots[i].wr.num_sge = 1;
    slots[i].wr.wr_id = i;
    qp.post_recv(&slots[i].wr);
  }

#ifdef _DEBUG
  cerr << "spip::IBVQueue::alocate qp.modify(IBV_QPS_RTR);" << endl;
#endif

  qp.modify(IBV_QPS_RTR);
}

void spip::IBVQueue::open (string ip_address, int port)
{
#ifdef _DEBUG
  cerr << "spip::IBVQueue::open ip=" << ip_address << " port=" << port << endl;
#endif
  boost::asio::ip::address_v4 interface_address = boost::asio::ip::address_v4::from_string(ip_address);

#ifdef _DEBUG
  cerr << "spip::IBVQueue::open build(" << interface_address << ")" << endl;
#endif
  build (interface_address);

  boost::asio::ip::udp::endpoint endpoint(interface_address, port);

#ifdef _DEBUG
  cerr << "spip::IBVQueue::open flows.push_back(create_flow(qp, " << endpoint << ", " << cm_id->port_num << ")" << endl;
#endif
  flows.push_back(create_flow(qp, endpoint, cm_id->port_num, false));
}

void spip::IBVQueue::open_multicast (string ip_address, string group, int port)
{
#ifdef _DEBUG
  cerr << "spip::IBVQueue::open_multicast ip_address=" << ip_address << " group=" << group << " port=" << port << endl;
#endif
  boost::asio::ip::address_v4 interface_address = boost::asio::ip::address_v4::from_string(ip_address);

  build (interface_address);

  // if we have the XXX.XXX.XXX.XXX+Y notation, then open a sequence of multicast groups
  std::string delimiter = "+";
  size_t pos = group.find(delimiter);

  // if no + notation
  if (pos == std::string::npos)
  {
    num_multicast = 1;
    groups.resize(num_multicast);
    groups[0] = group;
  }
  else
  {
    // get the XXX.XXX.XXX.XXX
    std::string mcast = group.substr(0, pos);
    // get the +Y
    std::string plus = group.substr(pos + delimiter.length());
    num_multicast = std::stoi (plus) + 1;

    // build multicast addresses
    groups.resize(num_multicast);

    std::string period = ".";
    size_t mcast_prefix_pos = mcast.find_last_of(period);
    std::string mcast_prefix = mcast.substr(0, mcast_prefix_pos);
    size_t mcast_suffix = std::stoi(mcast.substr(mcast_prefix_pos+1));

    for (unsigned i=0; i<num_multicast; i++)
    {
      groups[i] = mcast_prefix + "." + std::to_string(mcast_suffix + i);
    }
  }

#ifdef _DEBUG
  cerr << "spip::UDPSocketReceive::open_multicast num_multicast=" << num_multicast << endl;
#endif

#ifdef _TRACE
  for (unsigned i=0; i<num_multicast; i++)
    cerr << "spip::UDPSocketReceive::open_multicast group[" << i << "]=" << groups[i] << endl;
#endif

#ifdef _DEBUG
  cerr << "spip::UDPSocketReceive::open_multicast creating flows" << endl;
#endif
  for (unsigned i=0; i<num_multicast; i++)
  {
    boost::asio::ip::udp::endpoint endpoint(boost::asio::ip::address_v4::from_string(groups[i]), port);
    flows.push_back(create_flow(qp, endpoint, cm_id->port_num, true));
  }
}

void spip::IBVQueue::join_multicast (string ip_address, int port)
{
  boost::asio::ip::address_v4 interface_address = boost::asio::ip::address_v4::from_string(ip_address);

  // configure reuse address on the join socket
  join_socket.set_option(boost::asio::socket_base::reuse_address(true));

#ifdef _DEBUG
  cerr << "spip::UDPSocketReceive::join_multicast join MC groups" << endl;
#endif

  for (unsigned i=0; i<num_multicast; i++)
  {
    boost::asio::ip::udp::endpoint endpoint(boost::asio::ip::address_v4::from_string(groups[i]), port);
    join_socket.set_option(boost::asio::ip::multicast::join_group( endpoint.address().to_v4(), interface_address));
  }
}

int spip::IBVQueue::open_packet ()
{
  // if a slot is already open, return the packet
  if (islot != -1)
  {
    return payload.size();
  }

  // if we have processed all packets in the queue, get more
  if (ipacket == npackets)
  {
#ifdef _TRACE
    cerr << "spip::IBVQueue::open_packet polling receive queue for " << n_slots << " slots" << endl;
#endif
    npackets = cq.poll (n_slots, wc.get());
#ifdef _TRACE
    //if (npackets > 0)
      cerr << "spip::IBVQueue::open_packet recv_cq.poll returned " << npackets << " packets" << endl;
#endif
    ipacket = 0;
    islot = -1;
  }

  if (ipacket < npackets)
  {
    // work request ID indicates which slot the packet is present in
    islot = wc[ipacket].wr_id;

#ifdef _TRACE
    cerr << "spip::IBVQueue::open_packet processing packet " << ipacket << "/" << npackets << " in slot " << islot << endl;
#endif

    if (wc[ipacket].status != IBV_WC_SUCCESS)
    {
      cerr << "Work Request failed with code " <<  wc[ipacket].status << endl;
      sleep (1);
      islot = -1;
    }
    else
    {
#ifdef _TRACE
      cerr << "spip::IBVQueue::open_packet wc[" << ipacket << "].status == IBV_WC_SUCCESS" << endl;
#endif
      const void *ptr = reinterpret_cast<void *>(reinterpret_cast<std::uintptr_t>(slots[islot].sge.addr));
      std::size_t len = wc[ipacket].byte_len;

      // Sanity checks
      try
      {
#ifdef _TRACE
        cerr << "spip::IBVQueue::open_packet udp_from_ethernet()" << endl;
#endif
        payload = udp_from_ethernet (const_cast<void *>(ptr), len);

        buf_ptr = payload.data();
#ifdef _TRACE
        cerr << "spip::IBVQueue::open_packet received packet of size " << payload.size() << " bytes" << endl;
#endif
        return payload.size();
      }
      catch (packet_type_error &e)
      {
#ifdef _DEBUG
        cerr << "spip::IBVQueue::open_packet udp_from_ethernet failed: " << e.what() << endl;
#endif
        close_packet();
      }
    }
  }
  nsleeps++;

  // no packets available
  return 0;
}


void spip::IBVQueue::close_packet ()
{
#ifdef _TRACE
  cerr << "spip::IBVQueue::close_packet closing packet in slot " << islot << endl;
#endif

  if ((islot >= 0) && (islot < int(n_slots)))
  {
    qp.post_recv(&slots[islot].wr);

    // increment the packet counter
    ipacket++;

    // disable the slot index
    islot = -1;
  }
  else
  {
    throw runtime_error ("spip::IBVQueue::close_packet slot was not valid");
  }
}   


ibv_qp_t spip::IBVQueue::create_qp ( const ibv_pd_t &pd, 
    const ibv_cq_t &cq, std::size_t n_slots)
{
#ifdef _DEBUG
  cerr << "spip::IBVQueue::create_qp" << endl;
#endif
  ibv_qp_init_attr attr;
  memset(&attr, 0, sizeof(attr));
  attr.send_cq = cq.get();
  attr.recv_cq = cq.get();
  attr.qp_type = IBV_QPT_RAW_PACKET;
  attr.cap.max_send_wr = 1;
  attr.cap.max_recv_wr = n_slots;
  attr.cap.max_send_sge = 1;
  attr.cap.max_recv_sge = 1;
  return ibv_qp_t(pd, &attr);
}

ibv_flow_t spip::IBVQueue::create_flow(
    const ibv_qp_t &qp, const boost::asio::ip::udp::endpoint &endpoint, int port_num, bool mc)
{
#ifdef _TRACE
  cerr << "spip::IBVQueue::create_flow" << endl;
#endif
  struct
  {   
      ibv_flow_attr attr;
      ibv_flow_spec_eth eth;
      ibv_flow_spec_ipv4 ip;
      ibv_flow_spec_tcp_udp udp;
  } __attribute__((packed)) flow_rule;
  memset(&flow_rule, 0, sizeof(flow_rule));
  
  flow_rule.attr.type = IBV_FLOW_ATTR_NORMAL;
  flow_rule.attr.priority = 0;
  flow_rule.attr.size = sizeof(flow_rule);
  flow_rule.attr.num_of_specs = 3;
  flow_rule.attr.port = port_num;
  
  /* At least the ConnectX-3 cards seem to require an Ethernet match. We
   * thus have to construct the Ethernet multicast address corresponding to
   * the IP multicast address from RFC 7042.
   */
  flow_rule.eth.type = IBV_FLOW_SPEC_ETH;
  flow_rule.eth.size = sizeof(flow_rule.eth);
  if (mc)
  {
      spead2::mac_address dst_mac = multicast_mac(endpoint.address());
      std::memcpy(&flow_rule.eth.val.dst_mac, &dst_mac, sizeof(dst_mac));
  } 
  else
  {
      spead2::mac_address dst_mac = interface_mac(endpoint.address());
      std::memcpy(&flow_rule.eth.val.dst_mac, &dst_mac, sizeof(dst_mac));
  }
  // Set all 1's mask
  std::memset(&flow_rule.eth.mask.dst_mac, 0xFF, sizeof(flow_rule.eth.mask.dst_mac));

  flow_rule.ip.type = IBV_FLOW_SPEC_IPV4;
  flow_rule.ip.size = sizeof(flow_rule.ip);
  auto bytes = endpoint.address().to_v4().to_bytes(); // big-endian address
  std::memcpy(&flow_rule.ip.val.dst_ip, &bytes, sizeof(bytes));
  std::memset(&flow_rule.ip.mask.dst_ip, 0xFF, sizeof(flow_rule.ip.mask.dst_ip));
  
  flow_rule.udp.type = IBV_FLOW_SPEC_UDP;
  flow_rule.udp.size = sizeof(flow_rule.udp);
  flow_rule.udp.val.dst_port = htobe16(endpoint.port());
  flow_rule.udp.mask.dst_port = 0xFFFF;

#ifdef _TRACE
  cerr << "spip::IBVQueue::create_flow flow rule created" << endl;
#endif

  return ibv_flow_t(qp, &flow_rule.attr);
}
