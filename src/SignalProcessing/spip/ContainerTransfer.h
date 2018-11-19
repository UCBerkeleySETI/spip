//-*-C++-*-
/***************************************************************************
 *
 *   Copyright (C) 2018 by Andrew Jameson
 *   Licensed under the Academic Free License version 2.1
 *
 ***************************************************************************/

#ifndef __ContainerTransfer_h
#define __ContainerTransfer_h

#include "spip/Transformation.h"

namespace spip {

  template <class In, class Out>
  class ContainerTransfer : public Transformation <In, Out>
  {
    public:
     
      ContainerTransfer ();

      ~ContainerTransfer ();

      void set_output_reblock (unsigned f ) { nblock_out = f; };

    protected:

      unsigned iblock_out;

      unsigned nblock_out;

      uint64_t ndat;

    private:

  };

}

#endif
