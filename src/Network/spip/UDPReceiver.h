
#ifndef __UDPReceiver_h
#define __UDPReceiver_h

#include "spip/AsciiHeader.h"
#include "spip/UDPFormat.h"
#include "spip/UDPStats.h"

#include <cstdlib>

namespace spip {

  class UDPReceiver {

    public:

      UDPReceiver ();

      ~UDPReceiver ();

      void configure (const char * header_str);

      void prepare ();

      void set_format (UDPFormat * fmt);

      void stop_receiving ();

      // transmission thread
      void receive ();

      UDPStats * get_stats () { return stats; };

      char verbose;

    protected:

      UDPFormat * format;

      UDPStats * stats;

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
  };

}

#endif
