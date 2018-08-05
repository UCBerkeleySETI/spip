//-*-C++-*-
/***************************************************************************
 *
 *   Copyright (C) 2018 by Andrew Jameson
 *   Licensed under the Academic Free License version 2.1
 *
 ***************************************************************************/

#ifndef __SampleFold_h
#define __SampleFold_h

#include "spip/ContainerCUDA.h"
#include "spip/Transformation.h"

namespace spip {

  class SampleFold: public Transformation <Container, Container>
  {
    public:
     
      SampleFold ();

      ~SampleFold ();

      void configure (Ordering output_order);

      void set_periodicity (unsigned, uint64_t, uint64_t);

      void prepare ();

      void prepare_output ();

      void reserve ();

      //! Perform SampleFoldion on input block
      void transformation ();

      //! Required data transformations
      virtual void transform_SFPT_to_TSPFB () = 0;

      virtual void transform_TSPF_to_TSPFB () = 0;

      virtual void transform_TFPS_to_TSPFB () = 0;

    protected:

      Container * buffer;

      uint64_t buffer_idat;

      unsigned nchan;

      unsigned npol;

      unsigned ndim;

      unsigned nbit;

      unsigned nsignal;

      uint64_t ndat;

      unsigned nbin;

      double tsamp;

      unsigned nfolding_bin;

      uint64_t dat_offset;

      uint64_t dat_dec;

      Signal::State state;

    private:

  };

}

#endif
