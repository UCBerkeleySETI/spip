//-*-C++-*-
/***************************************************************************
 *
 *   Copyright (C) 2018 by Andrew Jameson
 *   Licensed under the Academic Free License version 2.1
 *
 ***************************************************************************/

#ifndef __RAMtoRAMTransfer_h
#define __RAMtoRAMTransfer_h

#include "spip/ContainerRAM.h"
#include "spip/Transformation.h"

namespace spip {

  class RAMtoRAMTransfer: public Transformation <Container, Container>
  {
    public:
     
      RAMtoRAMTransfer ();

      ~RAMtoRAMTransfer ();

      void set_output_reblock (unsigned);

      void configure (Ordering output_order);

      void prepare ();

      void prepare_output ();

      void reserve ();

      void transformation ();

    protected:

    private:

      unsigned iblock_out;

      unsigned nblock_out;

      uint64_t ndat;

  };

}

#endif
