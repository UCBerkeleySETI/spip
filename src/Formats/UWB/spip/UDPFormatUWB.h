/***************************************************************************
 *
 *   Copyright (C) 2016 Andrew Jameson
 *   Licensed under the Academic Free License version 2.1
 *
 ***************************************************************************/

#ifndef __UDPFormatUWB_h
#define __UDPFormatUWB_h

#include "vdifio.h"

#include "spip/UDPFormatVDIF.h"

#include <cstring>

namespace spip {

  class UDPFormatUWB : public UDPFormatVDIF {

    public:

      UDPFormatUWB (int pps = -1);

      ~UDPFormatUWB ();

      void configure (const spip::AsciiHeader& config, const char* suffix);

      uint64_t get_resolution ();

      uint64_t get_samples_for_bytes (uint64_t nbytes);

      inline int64_t decode_packet (char * buf, unsigned * payload_size);

      inline void gen_packet (char * buf, size_t bufsz);
   
    protected:

    private:

  };

}

#endif
