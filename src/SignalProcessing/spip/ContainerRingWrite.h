/***************************************************************************
 *
 *   Copyright (C) 2016 Andrew Jameson
 *   Licensed under the Academic Free License version 2.1
 *
 ***************************************************************************/

#include "config.h"
#include "spip/ContainerRing.h"
#include "spip/DataBlockWrite.h"

#ifndef __ContainerRingWrite_h
#define __ContainerRingWrite_h

namespace spip {

  class ContainerRingWrite : public ContainerRing
  {
    public:

      //! Null constructor
      ContainerRingWrite (DataBlockWrite *);

      ~ContainerRingWrite();

      void process_header ();

      void open_block ();

      void close_block ();

#ifdef HAVE_CUDA
      void register_buffers();
#endif

    protected:

    private:

      DataBlockWrite * db;

  };
}

#endif
