/***************************************************************************
 *
 *   Copyright (C) 2018 Andrew Jameson
 *   Licensed under the Academic Free License version 2.1
 *
 ***************************************************************************/

#ifndef __UDPFormatDualVDIF_h
#define __UDPFormatDualVDIF_h

#include "vdifio.h"

#include "spip/UDPFormatVDIF.h"

#include <cstring>

namespace spip {

  class UDPFormatDualVDIF : public UDPFormatVDIF {

    public:

      UDPFormatDualVDIF (int pps = -1);

      ~UDPFormatDualVDIF ();

      uint64_t get_resolution ();

      inline int64_t decode_packet (char * buf, unsigned * payload_size);

      void gen_packet (char * buf, size_t bufsz);


  };

}

#endif
