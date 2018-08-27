//-*-C++-*-
/***************************************************************************
 *
 *   Copyright (C) 2018 by Andrew Jameson
 *   Licensed under the Academic Free License version 2.1
 *
 ***************************************************************************/

#ifndef __PolSelect_h
#define __PolSelect_h

#include "spip/Container.h"
#include "spip/Transformation.h"

namespace spip {

  class PolSelect: public Transformation <Container, Container>
  {
    public:
     
      PolSelect ();

      ~PolSelect ();

      void set_pol_reduction (unsigned);

      void configure (Ordering output_order);

      void prepare ();

      void prepare_output ();

      //! Perform 
      void transformation ();

      //! Required data transformations 
      virtual void bypass () = 0;

      virtual void transform_TSPF () = 0;

      virtual void transform_SFPT () = 0;

    protected:

      unsigned nchan;

      unsigned npol;

      unsigned ndim;

      unsigned nbit;

      unsigned nsignal;

      uint64_t ndat;

      // number of output polarisations
      unsigned out_npol;

      // number of polarisations to drop
      unsigned delta_npol;

    private:

  };
}

#endif
