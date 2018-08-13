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

using namespace std;
using namespace spead2;

spip::IBVQueue::IBVQueue () : join_socket(owner.get_strand().get_io_service(), boost::asio::ip::udp::v4())
{
  buffer_size = 65536;
  packet_size = 9000;
  n_slots = 0;
}

spip::IBVQueue::~IBVQueue ()
{
}


void spip::IBVQueue::build (const boost::asio::ip::address_v4 interface_address)
{
  if (n_slots != 0)
    throw runtime_error ("Queue must be configured prior to building");

  n_slots = int(buffer_size / packet_size);

  cm_id = rdma_cm_id_t (event_channel, nullptr, RDMA_PS_UDP);
  cm_id.bind_addr(interface_address);

  // assume no completion channel
  recv_cq = ibv_cq_t(cm_id, n_slots, nullptr);
  send_cq = ibv_cq_t(cm_id, 1, nullptr);

  // create the protection domain
  pd = ibv_pd_t(cm_id);
  
  // create the queue pair
  qp = create_qp (pd, send_cq, recv_cq, n_slots);
  qp.modify (IBV_QPS_INIT, cm_id->port_num);
}

void spip::IBVQueue::configure (size_t _buffer_size, size_t _packet_size, size_t _header_size)
{
  if (n_slots != 0)
    throw runtime_error ("Queue cannot be configured after building"); 
  buffer_size = _buffer_size;
  packet_size = _packet_size;
  header_size = _header_size;
}

void spip::IBVQueue::allocate ()
{
  std::shared_ptr<mmap_allocator> allocator = std::make_shared<mmap_allocator>(0, true);
  buffer = allocator->allocate(buffer_size, nullptr);
  mr = ibv_mr_t(pd, buffer.get(), buffer_size, IBV_ACCESS_LOCAL_WRITE);
  slots.reset(new slot[n_slots]);
  wc.reset(new ibv_wc[n_slots]);
  const std::size_t max_raw_size = packet_size + header_size;
  for (std::size_t i = 0; i < n_slots; i++)
  {
    std::memset(&slots[i], 0, sizeof(slots[i]));
    slots[i].sge.addr = (uintptr_t) &buffer[i * max_raw_size];
    slots[i].sge.length = max_raw_size;
    slots[i].sge.lkey = mr->lkey;
    slots[i].wr.sg_list = &slots[i].sge;
    slots[i].wr.num_sge = 1;
    slots[i].wr.wr_id = i;
    qp.post_recv(&slots[i].wr);
  }
}

void spip::IBVQueue::open (string ip_address, int port)
{
  boost::asio::ip::address_v4 interface_address = boost::asio::ip::address_v4::from_string(ip_address);

  build (interface_address);

  boost::asio::ip::udp::endpoint endpoint(interface_address, port);

  flows.push_back(create_flow(qp, endpoint, cm_id->port_num));
}

void spip::IBVQueue::open_multicast (string ip_address, string group, int port)
{
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

#ifdef _DEBUG
   cerr << "spip::UDPSocketReceive::open_multicast num_multicast=" << num_multicast << endl;
#endif

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

  // configure reuse address on the join socket
  join_socket.set_option(boost::asio::socket_base::reuse_address(true));

  for (unsigned i=0; i<num_multicast; i++)
  {
    boost::asio::ip::udp::endpoint endpoint(boost::asio::ip::address_v4::from_string(groups[i]), port);
    flows.push_back(create_flow(qp, endpoint, cm_id->port_num));
    join_socket.set_option(boost::asio::ip::multicast::join_group( endpoint.address(), interface_address));
  }
}

ibv_qp_t spip::IBVQueue::create_qp ( const ibv_pd_t &pd, 
    const ibv_cq_t &send_cq, const ibv_cq_t &recv_cq, std::size_t n_slots)
{
  ibv_qp_init_attr attr;
  memset(&attr, 0, sizeof(attr));
  attr.send_cq = send_cq.get();
  attr.recv_cq = recv_cq.get();
  attr.qp_type = IBV_QPT_RAW_PACKET;
  attr.cap.max_send_wr = 1;
  attr.cap.max_recv_wr = n_slots;
  attr.cap.max_send_sge = 1;
  attr.cap.max_recv_sge = 1;
  return ibv_qp_t(pd, &attr);
}

ibv_flow_t spip::IBVQueue::create_flow(
    const ibv_qp_t &qp, const boost::asio::ip::udp::endpoint &endpoint, int port_num)
{
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
  spead2::mac_address dst_mac = multicast_mac(endpoint.address());
  std::memcpy(&flow_rule.eth.val.dst_mac, &dst_mac, sizeof(dst_mac));
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

  return ibv_flow_t(qp, &flow_rule.attr);
}
