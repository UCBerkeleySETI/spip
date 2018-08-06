//-*-C++-*-
/***************************************************************************
 *
 *   Copyright (C) 2017 by Andrew Jameson
 *   Licensed under the Academic Free License version 2.1
 *
 ***************************************************************************/

#ifndef __Append_h
#define __Append_h

#include "spip/Container.h"
#include "spip/Combination.h"

namespace spip {

  class Append: public Combination <Container, Container>
  {
    public:

      Append (const char * name);

      ~Append ();

      virtual void configure (Ordering output_order) = 0;

      virtual void set_output_state (Signal::State _state) = 0;

      void prepare ();

      void prepare_output ();

      void reserve ();

      //! Perform Append on input block
      void combination ();

      //! Required implementations
      virtual void combine_SFPT_to_SFPT () = 0;

      virtual void combine_TSPF_to_TSPF () = 0;

      virtual void combine_TFPS_to_TFPS () = 0;

      virtual void combine_TSPFB_to_TSPFB () = 0;

    protected:

      unsigned nchan;

      unsigned npol;

      unsigned ndim;

      unsigned nbit;

      unsigned nsignal;

      uint64_t ndat;

      uint64_t nbin;

      Signal::State state;

    private:

  };

}

#endif
