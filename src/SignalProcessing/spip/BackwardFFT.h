//-*-C++-*-
/***************************************************************************
 *
 *   Copyright (C) 2017 by Andrew Jameson
 *   Licensed under the Academic Free License version 2.1
 *
 ***************************************************************************/

#ifndef __BackwardFFT_h
#define __BackwardFFT_h

#include "spip/ContainerRAM.h"
#include "spip/Transformation.h"

namespace spip {

  class BackwardFFT: public Transformation <Container, Container>
  {
    public:
     
      BackwardFFT ();

      ~BackwardFFT ();

      void configure ();

      //! configuration of FFT plan
      virtual void configure_plan () = 0;

      void configure_plan_dimensions ();

      void prepare ();

      void prepare_output ();

      void reserve ();

      //! Perform a backward FFT on input block
      void transformation ();

      //! transform from TSPF input format
      virtual void transform_TSPF_to_SFPT () = 0 ;

      //! transform from TFPS input format
      virtual void transform_TFPS_to_SFPT () = 0 ;

      void set_nfft (int);

    protected:

      unsigned nchan;

      unsigned npol;

      unsigned ndim;

      unsigned nbit;

      unsigned nsignal;

      uint64_t ndat;

      int nfft;

      double tsamp;

      unsigned nbatch;

      unsigned nchan_out;

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
