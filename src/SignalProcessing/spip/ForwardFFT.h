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

      void configure ();

      void prepare ();

      void prepare_output ();

      void reserve ();

      //! Perform a forward FFT on input block
      void transformation ();

      //! Data transformation
      virtual void transform () = 0 ;

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

    private:

  };

}

#endif
