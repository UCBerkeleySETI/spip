//-*-C++-*-
/***************************************************************************
 *
 *   Copyright (C) 2018 by Andrew Jameson
 *   Licensed under the Academic Free License version 2.1
 *
 ***************************************************************************/

namespace spip {

  //! Defines various signal types
  namespace Signal {

    //! Possible states of the data
    enum State
    { 
      //! Nyquist sampled voltages (real)
      Nyquist,
      //! In-phase and Quadrature sampled voltages (complex)
      Analytic,
      //! Square-law detected total power
      Intensity,
      //! Square-law detected nth power
      NthPower,
      //! Square-law detected, two polarizations
      PPQQ,
      //! PP, QQ, Re[PQ], Im[PQ]
      Coherence,
      //! Stokes I,Q,U,V
      Stokes,
      //! PseudoStokes S0,S2,S2,S3
      PseudoStokes,
      //! Stokes invariant interval
      Invariant,
      //! Other
      Other,
      //! Just PP
      PP_State,
      //! Just QQ
      QQ_State,
      //! Fourth moment of the electric field (covariance of Stokes parameters)
      FourthMoment
    };
  }
}
