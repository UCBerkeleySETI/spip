/***************************************************************************
 *
 *   Copyright (C) 2016 Andrew Jameson
 *   Licensed under the Academic Free License version 2.1
 *
 ***************************************************************************/

#include "config.h"
#include "spip/ContainerRing.h"
#include "spip/DataBlockWrite.h"

#ifndef __ContainerRingWriteCUDA_h
#define __ContainerRingWriteCUDA_h

namespace spip {

  class ContainerRingWriteCUDA : public ContainerRing
  {
    public:

      //! Null constructor
      ContainerRingWriteCUDA (DataBlockWrite *);

      ~ContainerRingWriteCUDA();

      void process_header ();

      uint64_t open_block ();

      void close_block ();

      void zero ();

    protected:

    private:

      DataBlockWrite * db;

  };
}

#endif
