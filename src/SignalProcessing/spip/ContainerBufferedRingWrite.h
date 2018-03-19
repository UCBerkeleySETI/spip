/***************************************************************************
 *
 *   Copyright (C) 2018 Andrew Jameson
 *   Licensed under the Academic Free License version 2.1
 *
 ***************************************************************************/

#include "config.h"
#include "spip/ContainerBufferedRing.h"
#include "spip/DataBlockWrite.h"

#ifndef __ContainerBufferedRingWrite_h
#define __ContainerBufferedRingWrite_h

namespace spip {

  class ContainerBufferedRingWrite : public ContainerBufferedRing
  {
    public:

      //! Null constructor
      ContainerBufferedRingWrite (DataBlockWrite *);

      ~ContainerBufferedRingWrite();

      void process_header ();

      void write_buffer();

#ifdef HAVE_CUDA
      void register_buffers();
#endif

    protected:

    private:

      DataBlockWrite * db;

  };
}

#endif
