//-*-C++-*-
/***************************************************************************
 *
 *   Copyright (C) 2017 by Andrew Jameson
 *   Licensed under the Academic Free License version 2.1
 *
 ***************************************************************************/

#ifndef __AdaptiveFilter_h
#define __AdaptiveFilter_h

#include "spip/Container.h"
#include "spip/Transformation.h"

namespace spip {

  class AdaptiveFilter: public Transformation <Container, Container>
  {
    public:
     
      AdaptiveFilter ();

      ~AdaptiveFilter ();

      void configure (Ordering output_order);

      virtual void set_input_ref (Container *) = 0;

      void prepare ();

      void prepare_output ();

      void reserve ();

      //! Perform 
      void transformation ();

      //! Required data transformations 
      virtual void transform_TSPF () = 0;

      virtual void transform_SFPT () = 0;

    protected:

      Container * input_ref;

      unsigned nchan;

      unsigned npol;

      unsigned ndim;

      unsigned nbit;

      unsigned nsignal;

      uint64_t ndat;

      double tsamp;

      // generic container to store the gains
      Container * gains;

      // from Nuer's simulation and Adaptive Filter paper
      float epsilon;

      // number of samples to integrate into the filter
      unsigned filter_update_time;

    private:

  };

}

#endif
