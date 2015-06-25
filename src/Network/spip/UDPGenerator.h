
#ifndef __UDPGenerator_h
#define __UDPGenerator_h

#include "spip/UDPSocketSend.h"
#include "spip/UDPFormat.h"
#include "spip/UDPStats.h"

#include <cstdlib>

namespace spip {

  class UDPGenerator {

    public:

      UDPGenerator ();

      ~UDPGenerator ();

      int configure (const char * header);

      void allocate_signal ();

      void set_format (UDPFormat * format);

      void prepare (std::string ip_address, int port);

      // transmission thread
      void transmit (unsigned tobs, float data_rate);

      UDPStats * get_stats () { return stats; };

    protected:

      UDPSocketSend * sock;

      UDPFormat * format;

      UDPStats * stats;

      unsigned nchan;

      unsigned ndim;

      unsigned nbit;

      unsigned npol;

      float bw;

      float channel_bw;

      float tsamp;

      void * signal_buffer;

      size_t signal_buffer_size;

      unsigned bits_per_second;

      unsigned bytes_per_second;
  };

}

#endif
