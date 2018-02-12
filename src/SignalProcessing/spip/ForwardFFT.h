//-*-C++-*-
/***************************************************************************
 *
 *   Copyright (C) 2017 by Andrew Jameson
 *   Licensed under the Academic Free License version 2.1
 *
 ***************************************************************************/

#ifndef __ForwardFFT_h
#define __ForwardFFT_h

#include "spip/ContainerRAM.h"
#include "spip/Transformation.h"

namespace spip {

  class ForwardFFT: public Transformation <Container, Container>
  {
    public:
     
      ForwardFFT ();

      ~ForwardFFT ();

      void configure (Ordering output_order);

      //! configure the FFT plan
      virtual void configure_plan () = 0;

      void configure_plan_dimensions ();

      void prepare ();

      void prepare_output ();

      void reserve ();

      //! Perform a forward FFT on input block
      void transformation ();

      //! Required data transformation
      virtual void transform_SFPT_to_TFPS () = 0 ;

      //! Required data transformation
      virtual void transform_SFPT_to_TSPF () = 0 ;

      //! Required data transformation
      virtual void transform_SFPT_to_SFPT () = 0 ;

      void set_nfft (int);

    protected:

      unsigned nchan;

      unsigned npol;

      unsigned ndim;

      unsigned nbit;

      unsigned nsignal;

      uint64_t ndat;

      unsigned nbatch;

      unsigned nchan_out;

      int nfft;

      double tsamp;

      int rank;

      int n[1];

      int howmany;

      int inembed[1];

      int onembed[1];

      int istride;

      int idist;

      int ostride;

      int odist;

    private:

  };

}

#endif
