//-*-C++-*-
/***************************************************************************
 *
 *   Copyright (C) 2016 by Andrew Jameson
 *   Licensed under the Academic Free License version 2.1
 *
 ***************************************************************************/

#ifndef __IntegerDelay_h
#define __IntegerDelay_h

#include "spip/Container.h"
#include "spip/Transformation.h"

namespace spip {

  class IntegerDelay: public Transformation <Container, Container>
  {
    public:
     
      IntegerDelay ();

      ~IntegerDelay ();

      void configure (Ordering output_order);

      void prepare ();

      void prepare_output ();

      void reserve ();

      //! Set the integer delay for a specific channel
      void set_delay (unsigned isig, unsigned delay);

      //! Perform the delay of input to output
      void transformation ();

      //! Data transformation
      virtual void transform_SFPT_to_SFPT () = 0 ;

      void compute_delta_delays ();

      bool have_output () { return have_buffered_output; }

    protected:

    private:

      //! integer delays applied to previous block
      Container * prev_delays;

      //! integer delays to be applied to current block
      Container * curr_delays;

      //! difference between prev and curr delays
      Container * delta_delays;

      //! buffered output
      Container * buffered;

      bool have_buffered_output;

      unsigned nchan;

      unsigned npol;

      unsigned ndim;

      unsigned nbit;

      unsigned nsignal;

      uint64_t ndat;

  };

}

#endif
