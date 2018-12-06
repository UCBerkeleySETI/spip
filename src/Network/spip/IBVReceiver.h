
#ifndef __IBVReceiver_h
#define __IBVReceiver_h

#include "spip/AsciiHeader.h"
#include "spip/IBVQueue.h"
#include "spip/UDPFormat.h"
#include "spip/UDPStats.h"
#include "spip/UDPOverflow.h"

#include <cstdlib>

namespace spip {

  class IBVReceiver {

    public:

      IBVReceiver ();

      ~IBVReceiver ();

      void configure (const char * header_str);

      void prepare ();

      void set_format (UDPFormat * fmt);

      void stop_receiving ();

      // transmission thread
      void receive ();

      UDPStats * get_stats () { return stats; };

      char verbose;

    protected:

      void open_block ();

      bool process_packet (const std::uint8_t *data, std::size_t length);

      void close_block ();

      IBVQueue * queue;

      UDPFormat * format;

      UDPStats * stats;

      UDPOverflow * overflow;

      std::string data_host;

      std::string data_mcast;

      int data_port;

      AsciiHeader header;

      unsigned nchan;

      unsigned ndim;

      unsigned nbit;

      unsigned npol;

      float bw;

      float channel_bw;

      double tsamp;

      uint64_t bits_per_second;

      uint64_t bytes_per_second;

      bool keep_receiving;

      bool have_utc_start;

    private:

      bool need_next_block;

      bool filled_this_block;

      int64_t curr_byte_offset;

      int64_t next_byte_offset;

      int64_t last_byte_offset;

      char * curr_block;

      char * next_block;

      uint64_t data_bufsz;

      size_t packet_size;

      size_t header_size;

      size_t buffer_size;

      uint64_t bytes_curr_buf;

      uint64_t bytes_next_buf;

  };

}

#endif
