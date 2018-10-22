//-*-C++-*-
/***************************************************************************
 *
 *   Copyright (C) 2018 by Andrew Jameson
 *   Licensed under the Academic Free License version 2.1
 *
 ***************************************************************************/

#ifndef __PolCombine_h
#define __PolCombine_h

#include "spip/Container.h"
#include "spip/Combination.h"

namespace spip {

  class PolCombine: public Combination<Container, Container>
  {
    public:
     
      PolCombine ();

      ~PolCombine ();

      void configure (Ordering output_order);

      void prepare ();

      void prepare_output ();

      //! Perform 
      void combination();

      //! Required data combineations 
      virtual void combine_SFPT () = 0;

    protected:

      unsigned nchan;

      unsigned ndim;

      unsigned nbit;

      unsigned nsignal;

      uint64_t ndat;

      // number of output polarisations
      unsigned total_npol;

    private:

  };
}

#endif
