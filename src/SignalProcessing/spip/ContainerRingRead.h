/***************************************************************************
 *
 *   Copyright (C) 2016 Andrew Jameson
 *   Licensed under the Academic Free License version 2.1
 *
 ***************************************************************************/

#include "config.h"
#include "spip/ContainerRing.h"
#include "spip/DataBlockRead.h"

#ifndef __ContainerRingRead_h
#define __ContainerRingRead_h

namespace spip {

  class ContainerRingRead : public ContainerRing
  {
    public:

      //! Null constructor
      ContainerRingRead (DataBlockRead *);

      ~ContainerRingRead();

      void process_header ();

      uint64_t open_block ();

      void close_block ();

#ifdef HAVE_CUDA
      void register_buffers();

      int get_db_device () { return db->get_device(); };
      int get_db_device () const { return db->get_device(); };
#endif

    protected:

    private:

      DataBlockRead * db;

      uint64_t curr_buf_bytes;

  };
}

#endif
