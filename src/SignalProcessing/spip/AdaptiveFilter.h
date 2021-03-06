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
     
      AdaptiveFilter (const std::string&);

      ~AdaptiveFilter ();

      void set_filtering (int, double);

      void configure (Ordering output_order);
        
      void prepare ();

      void prepare_output ();

      void reserve ();

      unsigned get_blocks_per_mon_tsamp() { return blocks_per_mon_tsamp; };

      //! Perform 
      void transformation ();

      //! Required data transformations 
      virtual void transform_TSPF () = 0;

      virtual void transform_SFPT () = 0;

      //! Required transformation to write gain values to disk
      virtual void write_gains () = 0;

      virtual void write_dirty () = 0;

      virtual void write_cleaned () = 0;

    protected:

      std::string output_dir;

      unsigned nchan;

      unsigned npol;

      unsigned ndim;

      unsigned nbit;

      unsigned nsignal;

      uint64_t ndat;

      double tsamp;

      // generic container to store the gains
      Container * gains;

      // generic container to store the dirty
      Container * dirty;

      // generic container to store the cleaned
      Container * cleaned;

      // generic container to store the normalization factor
      Container * norms;

      // from Nuer's simulation and Adaptive Filter paper
      float epsilon;

      // number of samples to integrate into the filter
      unsigned filter_update_time;

      // number of output polarisations
      unsigned out_npol;

      // polarisation containing the RFI reference signal (<=0 means none)
      int ref_pol;

      // sampling time of monitoring output
      double req_mon_tsamp;

      double mon_tsamp;

      // number of input blocks per monitoring tsamp
      unsigned blocks_per_mon_tsamp;

      bool perform_filtering;

    private:

  };

}

#endif
