//-*-C++-*-
/***************************************************************************
 *
 *   Copyright (C) 2017 by Andrew Jameson
 *   Licensed under the Academic Free License version 2.1
 *
 ***************************************************************************/

#ifndef __UnpackFloat_h
#define __UnpackFloat_h

#include "spip/Container.h"
#include "spip/Transformation.h"

namespace spip {

  class UnpackFloat: public Transformation <Container, Container>
  {
    public:
     
      UnpackFloat ();

      ~UnpackFloat ();

      void configure (Ordering output_order);

      void prepare ();

      void prepare_output ();

      void reserve ();

      //! Perform a forward FFT on input block
      void transformation ();

      //! Data transformation
      virtual void transform_SFPT_to_SFPT () = 0 ;

    protected:

      Endian endianness;

      Encoding encoding;

      // hacks to accomodate nvcc and enums
      bool big_endian;

      bool twos_complement;

      unsigned nchan;

      unsigned npol;

      unsigned ndim;

      unsigned nbit;

      unsigned nsignal;

      uint64_t ndat;

      float offset;

      float scale;

    private:

  };

}

#endif
