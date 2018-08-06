//-*-C++-*-
/***************************************************************************
 *
 *   Copyright (C) 2018 by Andrew Jameson
 *   Licensed under the Academic Free License version 2.1
 *
 ***************************************************************************/

#ifndef __IntegrationBinned_h
#define __IntegrationBinned_h

#include "spip/Container.h"
#include "spip/Transformation.h"
#include "spip/Time.h"

namespace spip {

  class IntegrationBinned: public Transformation <Container, Container>
  {
    public:
     
      IntegrationBinned ();

      ~IntegrationBinned ();

      void configure (Ordering output_order);

      void set_decimation (unsigned, unsigned, unsigned, unsigned);

      void prepare ();

      void prepare_output ();

      void prepare_binplan ();

      void reserve ();

      //! Perform Integration and Binning on input block
      void transformation ();

      //! Required data transformations
      virtual void transform_TSPF_to_TSPFB () = 0;

    protected:

      Container * buffer;

      Container * fscr;

      std::vector<int> binplan;

      int64_t buffer_idat;

      unsigned nchan;

      unsigned npol;

      unsigned ndim;

      unsigned nbit;

      unsigned nsignal;

      unsigned nbin;

      uint64_t ndat;

      unsigned dat_dec;

      unsigned chan_dec;

      unsigned pol_dec;

      unsigned signal_dec;

      double tsamp;

      Signal::State state;

    private:

      int cal_signal;
    
      double cal_freq;

      double cal_period;

      double cal_phase;

      double cal_duty_cycle;

      Time * cal_epoch;

      int64_t cal_epoch_delta;

      uint64_t start_idat;
  };

}

#endif
