
#ifndef __IBVSocket_h
#define __IBVSocket_h

#ifndef _GNU_SOURCE
#define _GNU_SOURCE
#endif


#include <spead2/common_ibv.h>
#include <spead2/common_memory_allocator.h>
#include <spead2/common_raw_packet.h>

#include <boost/asio.hpp>
#include <boost/noncopyable.hpp>

#include <string>

namespace spip {

  class IBVQueue {

    public:

      //! Global flag to cease receiving
      static bool keep_receiving;

      IBVQueue (boost::asio::io_service& io_service);

      ~IBVQueue ();

      // open the socket
      void build (const boost::asio::ip::address_v4 interface_address);

      void allocate ();

      void configure (size_t buffer_size, size_t packet_size, size_t header_size);

      void open (std::string ip_address, int port);

      void open_multicast (std::string ip_address, std::string group, int port);

      void poll_once ();

      int open_packet ();

      void close_packet ();
       
      uint8_t * buf_ptr;

    protected:

      // structure for IBV socket
      struct sockaddr_in udp_sock;

      // for other end-point of IBV socket
      struct sockaddr_in other_udp_sock;

    private:

      // Socket that is used only to join the multicast group. It is not bound to a port 
      boost::asio::ip::udp::socket join_socket;

      spead2::memory_allocator::pointer buffer;

      size_t buffer_size;

      size_t packet_size;

      size_t header_size;

      size_t max_raw_size;

      size_t n_slots;

      size_t num_multicast;

      std::vector<std::string> groups;

      struct slot : boost::noncopyable
      {
        ibv_recv_wr wr;
        ibv_sge sge;
      };

      // All the data structures required by ibverbs
      spead2::rdma_event_channel_t event_channel;
      spead2::rdma_cm_id_t cm_id;
      spead2::ibv_pd_t pd;
      spead2::ibv_comp_channel_t comp_channel;
      
      spead2::ibv_qp_t qp;
      std::vector<spead2::ibv_flow_t> flows;
      spead2::ibv_mr_t mr;

      spead2::ibv_cq_t cq;

      // array of @ref n_slots slots for work requests
      std::unique_ptr<slot[]> slots;

      // array of @ref n_slots work completions
      std::unique_ptr<ibv_wc[]> wc;

      int islot;

      int ipacket;

      int npackets;

      spead2::packet_buffer payload;

      // Utility functions to create the data structures
      static spead2::ibv_qp_t
      create_qp(const spead2::ibv_pd_t &pd, const spead2::ibv_cq_t &cq,
                std::size_t n_slots);

      static spead2::ibv_flow_t
      create_flow(const spead2::ibv_qp_t &qp, const boost::asio::ip::udp::endpoint &endpoint,
                  int port_num);

      static void req_notify_cq (ibv_cq *cq);

  };

}

#endif
