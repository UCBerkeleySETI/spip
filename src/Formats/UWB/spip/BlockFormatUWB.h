
#ifndef __BlockFormatUWB_h
#define __BlockFormatUWB_h

#include "spip/BlockFormat.h"
#include "config.h"

#include <cstring>

#ifdef HAVE_FFTW3
#include <fftw3.h>
#endif

namespace spip {

  class BlockFormatUWB : public BlockFormat {

    public:

      BlockFormatUWB (unsigned);

      ~BlockFormatUWB ();

#ifdef HAVE_FFTW3
      void deconfigure_fft ();

      void configure_fft ();
#endif

      void unpack_hgft (char * buffer, uint64_t nbytes);

      void unpack_ms (char * buffer, uint64_t nbytes);

    private:

      inline int16_t convert_offset_binary (int16_t in) { if (in == 0) return 0; else return in^0x8000; };

#ifdef HAVE_FFTW3
      fftwf_plan plan;

      unsigned nfft;

      fftwf_complex * in;

      fftwf_complex * out;
#endif

  };

}

#endif
