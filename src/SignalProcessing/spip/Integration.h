//-*-C++-*-
/***************************************************************************
 *
 *   Copyright (C) 2018 by Andrew Jameson
 *   Licensed under the Academic Free License version 2.1
 *
 ***************************************************************************/

#ifndef __Integration_h
#define __Integration_h

#include "spip/Container.h"
#include "spip/Transformation.h"

namespace spip {

  class Integration: public Transformation <Container, Container>
  {
    public:
     
      Integration ();

      ~Integration ();

      void configure (Ordering output_order);

      void set_decimation (unsigned, unsigned, unsigned, unsigned);

      void prepare ();

      void prepare_output ();

      void reserve ();

      //! Perform Integrationion on input block
      void transformation ();

      //! Required data transformations
      //virtual void transform_SFPT_to_SFPT () = 0;

      virtual void transform_TSPF_to_TSPF () = 0;

      virtual void transform_TFPS_to_TFPS () = 0;


    protected:

      Container * buffer;

      Container * fscr;

      int64_t buffer_idat;

      unsigned nchan;

      unsigned npol;

      unsigned ndim;

      unsigned nbit;

      unsigned nsignal;

      uint64_t ndat;

      unsigned dat_dec;

      unsigned chan_dec;

      unsigned pol_dec;

      unsigned signal_dec;

      Signal::State state;

    private:

  };

}

#endif
