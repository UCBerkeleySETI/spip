
#ifndef __UDPOverflow_h
#define __UDPOverflow_h

#include <cstdlib>
#include <inttypes.h>

namespace spip {

  class UDPOverflow {

    public:

      UDPOverflow();

      virtual ~UDPOverflow();

      //! resize the buffer to the required number of bytes
      void resize (size_t);

      //! reset the overflow block
      void reset ();

      //! copy from the supplied buffer to overflow buffer
      void copy_from (char *, size_t, size_t);

      //! update overflow buffer counters based on remote access
      void copied_from (size_t, size_t);

      //! copy to the supplied buffer from the overflow buffer
      int64_t copy_to (char *);

      //! return size of the buffer
      size_t get_bufsz() { return bufsz; };

      //! return pointer to buffer for direct access
      char * get_buffer() { return buffer; };

      //! return the last byte copied
      size_t get_last_byte () { return last_byte; };

    protected:

      //! buffer 
      char * buffer;

      size_t bufsz;

      size_t last_byte;

      size_t overflowed_bytes;

    private:

  };

}

#endif
