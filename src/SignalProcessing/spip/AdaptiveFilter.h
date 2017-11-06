//-*-C++-*-
/***************************************************************************
 *
 *   Copyright (C) 2017 by Andrew Jameson
 *   Licensed under the Academic Free License version 2.1
 *
 ***************************************************************************/

#ifndef __AdaptiveFilter_h
#define __AdaptiveFilter_h

#include "spip/ContainerRAM.h"
#include "spip/Transformation.h"

namespace spip {

  class AdaptiveFilter: public Transformation <Container, Container>
  {
    public:
     
      AdaptiveFilter ();

      ~AdaptiveFilter ();

      void configure ();

      //virtual void set_rfi_input (Container) = 0;

      void prepare ();

      void prepare_output ();

      void reserve ();

      //! Perform 
      void transformation ();

      //! Required data transformation
      virtual void transform_TSPF () = 0;

    protected:

      unsigned nchan;

      unsigned npol;

      unsigned ndim;

      unsigned nbit;

      unsigned nsignal;

      uint64_t ndat;

      double tsamp;

    private:

  };

}

#endif
