//-*-C++-*-
/***************************************************************************
 *
 *   Copyright (C) 2017 by Andrew Jameson
 *   Licensed under the Academic Free License version 2.1
 *
 ***************************************************************************/

#ifndef __Detection_h
#define __Detection_h

#include "spip/Container.h"
#include "spip/Transformation.h"

namespace spip {

  class Detection: public Transformation <Container, Container>
  {
    public:

      Detection (const char * name);

      ~Detection ();

      virtual void configure (Ordering output_order) = 0;

      virtual void set_output_state (Signal::State _state) = 0;

      void prepare ();

      void prepare_output ();

      void reserve ();

      //! Perform Detection on input block
      void transformation ();

      //! Required implementstations
      virtual void transform_SFPT_to_SFPT () = 0;

      virtual void transform_TSPF_to_TSPF () = 0;

      virtual void transform_TFPS_to_TFPS () = 0;

    protected:

      unsigned nchan;

      unsigned npol;

      unsigned ndim;

      unsigned nbit;

      unsigned nsignal;

      uint64_t ndat;

      Signal::State state;

    private:

  };

}

#endif
