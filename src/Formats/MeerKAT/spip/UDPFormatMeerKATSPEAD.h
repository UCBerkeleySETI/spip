
#ifndef __UDPFormatMeerKATSPEAD_h
#define __UDPFormatMeerKATSPEAD_h

#include "spip/meerkat_def.h"
#include "spip/UDPFormat.h"
#include "spip/AsciiHeader.h"

#include "spead2/recv_packet.h"
#include "spead2/recv_udp.h"

#define UDP_FORMAT_MEERKAT_SPEAD_NDIM 2
#define UDP_FORMAT_MEERKAT_SPEAD_NPOL 1

#include <cstring>

static uint16_t magic_version = 0x5304;  // 0x53 is the magic, 4 is the version

namespace spip {

#ifndef OLD_DECODER
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
#endif

  class UDPFormatMeerKATSPEAD : public UDPFormat {

    public:

      UDPFormatMeerKATSPEAD ();

      ~UDPFormatMeerKATSPEAD ();

      void configure (const spip::AsciiHeader& config, const char* suffix);

      void prepare (spip::AsciiHeader& header, const char* suffix);

      void conclude ();

      void generate_signal ();

      uint64_t get_samples_for_bytes (uint64_t nbytes);

      uint64_t get_resolution ();

      void set_channel_range (unsigned start, unsigned end);

      int64_t get_timestamp_fast ();

      int64_t get_timestamp_and_channel();

      static void encode_seq (char * buf, uint64_t seq)
      {
        memcpy (buf, (void *) &seq, sizeof(uint64_t));
      };

      inline void encode_header_seq (char * buf, uint64_t packet_number);
      inline void encode_header (char * buf);

      void decode_spead (char * buf);

#ifdef OLD_DECODER
      void set_heap_num (int64_t heap_num ) { header.heap_cnt = heap_num * 8192; };
      void print_packet_timestamp ();
#else
      std::size_t decode_cbf_packet (cbf_packet_header &out, const uint8_t *data, std::size_t max_size);
#endif
      inline int64_t decode_packet (char * buf, unsigned *payload_size);
      inline int64_t get_subband (int64_t byte_offset, int nsubband);
      inline int insert_last_packet (char * buf);

      void print_packet_header ();
      bool check_stream_stop ();

      inline void gen_packet (char * buf, size_t bufsz);

      // accessor methods for header params
      void set_chan_no (int16_t chan_no)    { ; };
      void set_beam_no (int16_t beam_no)    { ; };

      static unsigned get_samples_per_packet () { return 1; };

    private:

#ifdef OLD_DECODER
      spead2::recv::packet_header header;
#else
      spip::cbf_packet_header cbf_header;
#endif

      time_t adc_sync_time;

      uint64_t adc_sample_rate;

      uint64_t adc_samples_per_heap;

      int64_t obs_start_sample;

      double samples_to_byte_offset;

      double bw;

      double tsamp;

      double adc_to_cbf;

      uint64_t nsamp_per_sec;

      unsigned nsamp_per_heap;

      unsigned nbytes_per_samp;

      unsigned avg_pkt_size;

      unsigned heap_size;

      unsigned pkts_per_heap;

      std::vector<int64_t> timestamps;

      std::vector<int64_t> channels;

      std::vector<int64_t> curr_heap_cnts;

      std::vector<uint64_t> curr_heap_offsets;

      unsigned nbytes_per_heap;

      unsigned timestamp_to_samples;

      bool first_heap;

      bool first_packet;

      unsigned header_npol;

      int offset ;

      int num_spead_streams;

      int channels_per_spead_stream;

      int half_num_spead_streams;

      int spead_stream;

  };


}

#endif
