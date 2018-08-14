
#ifndef __meerkat_spead_def_h
#define __meerkat_spead_def_h

#include "spead2/recv_packet.h"
#include "spead2/recv_udp.h"

static uint16_t magic_version = 0x5304;  // 0x53 is the magic, 4 is the version

namespace spip {

  struct cbf_packet_header
  {
    /// Number of bits in addresses/immediates (from SPEAD flavour)
    int heap_address_bits;
    /// Number of item pointers in the packet
    int n_items;
    /**
     * @name Key fields extracted from items in the packet
     * @{
     * The true values are always non-negative, and -1 is used to indicate
     * that the packet did not contain the item.
     */
    spead2::s_item_pointer_t heap_cnt;
    spead2::s_item_pointer_t heap_length;
    spead2::s_item_pointer_t payload_offset;
    spead2::s_item_pointer_t payload_length;
    spead2::s_item_pointer_t timestamp;
    spead2::s_item_pointer_t channel;
    /** @} */
    /// The item pointers in the packet, in big endian, and not necessarily aligned
    const std::uint8_t *pointers;
    /// Start of the packet payload
    const std::uint8_t *payload;
  };
}

#endif
