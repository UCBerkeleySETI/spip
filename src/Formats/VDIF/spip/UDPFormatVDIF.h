/***************************************************************************
 *
 *   Copyright (C) 2016 Andrew Jameson
 *   Licensed under the Academic Free License version 2.1
 *
 ***************************************************************************/

#ifndef __UDPFormatVDIF_h
#define __UDPFormatVDIF_h

#include "vdifio.h"

#include "spip/UDPFormat.h"

#include <cstring>

namespace spip {

  class UDPFormatVDIF : public UDPFormat {

    public:

      UDPFormatVDIF (int pps = -1);

      ~UDPFormatVDIF ();

      void configure (const AsciiHeader& config, const char* suffix);

      void prepare (AsciiHeader& header, const char* suffix);

      void configure_stream (char * buf);

      void conclude ();

      unsigned get_samples_per_packet () { return nsamp_per_packet; };

      void generate_signal ();

      uint64_t get_samples_for_bytes (uint64_t nbytes);

      uint64_t get_resolution ();

      void set_channel_range (unsigned start, unsigned end);

      void encode_header_seq (char * buf, uint64_t packet_number);
      void encode_header (char * buf);

      inline int64_t decode_packet (char * buf, unsigned * payload_size);

      inline uint64_t decode_header_seq (char * buf);
      inline void decode_header (char * buf);

      int insert_last_packet (char * buf);

      void print_packet_header ();

      inline void gen_packet (char * buf, size_t bufsz);
      
      //! encode header with config data
      void compute_header ();

      inline int64_t get_subband (int64_t byte_offset, int nsubband) { return 0; };

    protected:

      vdif_header header;

      char * payload;

      uint64_t nsamp_offset;

      unsigned packets_per_second;

      unsigned header_npol;

      uint64_t bytes_per_second;

      //! second (since VDIF epoch) that the observation began
      int start_second;

      unsigned offset;

      //! size of a single output data frame
      uint64_t frame_size;

      //! offset of this VDIF frame within the output data stream
      uint64_t frame_offset;

      bool configured_stream;

    private:

      unsigned nsamp_per_packet;

      unsigned udp_nsamp;

      double bw;

      double tsamp;

      int thread_id;

  };

}

#endif
