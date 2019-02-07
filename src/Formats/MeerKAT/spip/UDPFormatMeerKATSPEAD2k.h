
#ifndef __UDPFormatMeerKATSPEAD2k_h
#define __UDPFormatMeerKATSPEAD2k_h

#include "spip/meerkat_def.h"
#include "spip/meerkat_spead_def.h"
#include "spip/UDPFormat.h"
#include "spip/AsciiHeader.h"

#define UDP_FORMAT_MEERKAT_SPEAD_NDIM 2
#define UDP_FORMAT_MEERKAT_SPEAD_NPOL 1

#include <cstring>

namespace spip {

  class UDPFormatMeerKATSPEAD2k : public UDPFormat {

    public:

      UDPFormatMeerKATSPEAD2k ();

      ~UDPFormatMeerKATSPEAD2k ();

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

      std::size_t decode_cbf_packet (cbf_packet_header &out, const uint8_t *data, std::size_t max_size);
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

      spip::cbf_packet_header cbf_header;

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

      unsigned nbytes_per_heap;

      unsigned timestamp_to_samples;

      bool first_heap;

      bool first_packet;

      unsigned header_npol;

      int offset ;

      int nstream;

      int nchan_per_stream;

      int half_nstream;

      int spead_stream;

  };


}

#endif
