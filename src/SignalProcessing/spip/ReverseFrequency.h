//-*-C++-*-
/***************************************************************************
 *
 *   Copyright (C) 2018 by Andrew Jameson
 *   Licensed under the Academic Free License version 2.1
 *
 ***************************************************************************/

#ifndef __ReverseFrequency_h
#define __ReverseFrequency_h

#include "spip/Container.h"
#include "spip/Transformation.h"

namespace spip {

  class ReverseFrequency: public Transformation <Container, Container>
  {
    public:
     
      ReverseFrequency ();

      ~ReverseFrequency ();

      void configure (Ordering output_order);

      void set_sideband (Signal::Sideband);

      void prepare ();

      void prepare_output ();

      void reserve ();

      //! Perform ReverseFrequencyion on input block
      void transformation ();

      //! Required data transformations
      //virtual void transform_SFPT_to_SFPT () = 0;

      virtual void transform_TSPF_to_TSPF () = 0;

      virtual void transform_TFPS_to_TFPS () = 0;

    protected:

      unsigned nchan;

      unsigned npol;

      unsigned ndim;

      unsigned nbit;

      unsigned nsignal;

      uint64_t ndat;

      Signal::Sideband output_sideband;

      bool reversal;

    private:

  };

}

#endif
