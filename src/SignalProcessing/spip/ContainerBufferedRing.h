/***************************************************************************
 *
 *   Copyright (C) 2016 Andrew Jameson
 *   Licensed under the Academic Free License version 2.1
 *
 ***************************************************************************/
#include "config.h"
#include "spip/Container.h"
#include "spip/DataBlockRead.h"
#include "spip/DataBlockWrite.h"

#ifndef __ContainerBufferedRing_h
#define __ContainerBufferedRing_h

namespace spip {

  class ContainerBufferedRing : public Container
  {
    public:

      //! Null constructor
      ContainerBufferedRing ();

      ~ContainerBufferedRing();

      virtual void process_header () = 0;

      //! resize the buffer to match the input dimensions
      void resize ();

      //! zero the buffer 
      void zero ();

      //! set change the buffer pointer
      void set_buffer (unsigned char * buf);

      //! unset the buffer pointer
      void unset_buffer ();

      //! query whether the buffer is valid
      bool is_valid () { return buffer_valid; }

    protected:

    private:

      bool buffer_valid;

  };
}

#endif
