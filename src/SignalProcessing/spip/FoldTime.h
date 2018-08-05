//-*-C++-*-
/***************************************************************************
 *
 *   Copyright (C) 2018 by Andrew Jameson
 *   Licensed under the Academic Free License version 2.1
 *
 ***************************************************************************/

#ifndef __FoldTime_h
#define __FoldTime_h

#include "spip/Container.h"
#include "spip/Transformation.h"

namespace spip {

  class FoldTime: public Transformation <Container, Container>
  {
    public:
     
      FoldTime ();

      ~FoldTime ();

      void configure (Ordering output_order);

      void set_binning (unsigned, bool);

      void set_periodicity (double, double, double, uint64_t);

      void prepare ();

      void prepare_output ();

      void reserve ();

      //! Perform FoldTimeion on input block
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

      bool partial_bins;

      double period;

      double epoch;

      double duty_cycle;

      uint64_t dat_dec;

      Signal::State state;

    private:

  };

}

#endif
