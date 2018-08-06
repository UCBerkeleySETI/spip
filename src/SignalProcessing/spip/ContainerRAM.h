/***************************************************************************
 *
 *   Copyright (C) 2016 Andrew Jameson
 *   Licensed under the Academic Free License version 2.1
 *
 ***************************************************************************/

#include "spip/Container.h"

#ifndef __ContainerRAM_h
#define __ContainerRAM_h

namespace spip {

  class ContainerRAM : public Container
  {
    public:

      //! Null constructor
      ContainerRAM ();

      ~ContainerRAM();

      //! free the buffer, if allocated
      void free_buffer();

      //! resize the buffer to match the input dimensions
      void resize ();

      //! zero the buffer 
      void zero ();

      //! register the buffer as Pinned host memory for CUDA operations
      void register_buffer ();

      //! unregister the buffer as Pinned host memory for CUDA operations
      void unregister_buffer ();

    protected:

      bool buffer_registered;

      bool buffer_should_register;

    private:

  };
}

#endif
