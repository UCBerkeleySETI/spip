/***************************************************************************
 *
 *   Copyright (C) 2016 Andrew Jameson
 *   Licensed under the Academic Free License version 2.1
 *
 ***************************************************************************/

#include "config.h"
#include "spip/ContainerRing.h"
#include "spip/DataBlockRead.h"

#ifndef __ContainerRingReadCUDA_h
#define __ContainerRingReadCUDA_h

namespace spip {

  class ContainerRingReadCUDA : public ContainerRing
  {
    public:

      //! Null constructor
      ContainerRingReadCUDA (DataBlockRead *);

      ~ContainerRingReadCUDA();

      void process_header ();

      uint64_t open_block ();

      void close_block ();

      int get_db_device () { return db->get_device(); };
      int get_db_device () const { return db->get_device(); };

    protected:

    private:

      DataBlockRead * db;

      uint64_t nbits_per_sample;

      uint64_t curr_buf_bytes;

    protected:

    private:

  };
}

#endif
