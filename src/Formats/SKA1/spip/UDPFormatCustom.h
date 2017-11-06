
#ifndef __UDPFormatCustom_h
#define __UDPFormatCustom_h

#include "spip/ska1_def.h"
#include "spip/UDPFormat.h"

#define UDP_FORMAT_CUSTOM_PACKET_NSAMP 2048
#define UDP_FORMAT_CUSTOM_NDIM 2
#define UDP_FORMAT_CUSTOM_NPOL 2

#include <cstring>

namespace spip {

  typedef struct {

    uint64_t packet_sequence_number;

    uint64_t attoseconds_from_integer;

    uint32_t seconds_from_epoch;

    uint32_t cfcbi_number;	// Contiguous Frequnecy Channel Block Index

    uint16_t beam_number;

    uint8_t  cbf_version;

    uint8_t  reserved_01;
    uint8_t  reserved_02;
    uint8_t  reserved_03;
    uint8_t  reserved_04;
    uint8_t  reserved_05;

  } ska1_custom_udp_header_t;


  class UDPFormatCustom : public UDPFormat {

    public:

      UDPFormatCustom ();

      ~UDPFormatCustom ();

      void configure (const spip::AsciiHeader& config, const char* suffix);

      void prepare (spip::AsciiHeader& header, const char* suffix);

      void conclude () { ; } ;

      void generate_signal ();

      uint64_t get_samples_for_bytes (uint64_t nbytes);

      uint64_t get_resolution ();

      static void encode_seq (char * buf, uint64_t seq)
      {
        memcpy (buf, (void *) &seq, sizeof(uint64_t));
      };

      static inline uint64_t decode_seq (char * buf)
      {
        return ((uint64_t *) buf)[0];  
      };

      inline void encode_header_seq (char * buf, uint64_t packet_number);
      inline void encode_header (char * buf);

      inline uint64_t decode_header_seq (char * buf);
      inline unsigned decode_header (char * buf);

      inline int64_t decode_packet (char * buf, unsigned * payload_size);
      inline int insert_last_packet (char * buf);

      inline int check_packet ();
      inline int insert_packet (char * buf, char * pkt, uint64_t start_samp, uint64_t next_samp);

      void print_packet_header ();

      inline void gen_packet (char * buf, size_t bufsz);

      // accessor methods for header params
      void set_seq_num  (uint64_t seq_num)  { header.packet_sequence_number = seq_num; };
      void set_int_sec  (uint32_t int_sec)  { header.seconds_from_epoch = int_sec; };
      void set_atto_sec (uint64_t atto_sec) { header.attoseconds_from_integer = atto_sec; };
      void set_beam_num (uint16_t beam_num)  { header.beam_number = beam_num; };
      void set_cfcbi_num (uint16_t cfcbi_num) { header.cfcbi_number = cfcbi_num; };
      void set_cbf_ver (uint8_t  cbf_ver)   { header.cbf_version = cbf_ver; };
 
      static unsigned get_samples_per_packet () { return UDP_FORMAT_CUSTOM_PACKET_NSAMP; };

    private:

      ska1_custom_udp_header_t header;

      ska1_custom_udp_header_t * header_ptr;

      unsigned packet_weights_size;
     
      char * weights_ptr;

      char * payload_ptr;

      unsigned seq_stride;

      unsigned cfcbi_stride;

      uint64_t nsamp_per_sec;

      unsigned start_channel;

      unsigned end_channel;

      unsigned nsamp_per_weight;

      unsigned nsamp_per_packet;

      unsigned nchan_per_packet;

      unsigned start_cfcbi;

      unsigned end_cfcbi;

      unsigned ncfcbi;

      uint64_t attoseconds_per_packet;

  };

}

#endif
