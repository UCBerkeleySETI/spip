//-*-C++-*-
/***************************************************************************
 *
 *   Copyright (C) 2017 by Andrew Jameson
 *   Licensed under the Academic Free License version 2.1
 *
 ***************************************************************************/

#ifndef __BatchedBackwardFFT_h
#define __BatchedBackwardFFT_h

#include "spip/ContainerRAM.h"
#include "spip/Transformation.h"

namespace spip {

  class BatchedBackwardFFT: public Transformation <Container, Container>
  {
    public:
     
      BatchedBackwardFFT ();

      ~BatchedBackwardFFT ();

      void configure (Ordering output_order);

      void prepare ();

      void prepare_output ();

      void reserve ();

      //! Perform a forward FFT on input block
      void transformation ();

      //! Data transformation
      virtual void transform () = 0 ;

      void set_nfft (int _nfft) { nfft = _nfft; };

    protected:

      unsigned nchan;

      unsigned npol;

      unsigned ndim;

      unsigned nbit;

      unsigned nsignal;

      uint64_t ndat;

      double tsamp;

      int nfft;

      uint64_t ndat_out;

      unsigned nchan_out;

      double tsamp_out;

    private:

  };

}

#endif
