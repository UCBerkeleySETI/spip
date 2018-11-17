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

      void reserve ();

      //! Perform Integration and Binning on input block
      void transformation ();

      //! Required preparation
      virtual void prepare_binplan () = 0;

      //! Required data transformations
      virtual void transform_TSPF_to_TSPFB () = 0;

      size_t get_sample_size ();

    protected:

      Container * buffer;

      Container * fscr;

      Container * binplan;
      //std::vector<int> binplan;

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

      int cal_signal;
    
      double cal_period;

      double cal_phase;

      double cal_duty_cycle;

      int64_t cal_epoch_delta;

      uint64_t start_idat;

    private:

      double cal_freq;

      Time * cal_epoch;

  };

}

#endif
