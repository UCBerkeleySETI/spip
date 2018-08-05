/***************************************************************************
 *
 *   Copyright (C) 2016 Andrew Jameson
 *   Licensed under the Academic Free License version 2.1
 *
 ***************************************************************************/

#include "spip/DetectionPolarimetryCUDA.h"

#include <iostream>
#include <cstring>
#include <stdexcept>
#include <cmath>
#include <cuComplex.h>

using namespace std;

spip::DetectionPolarimetryCUDA::DetectionPolarimetryCUDA (cudaStream_t _stream) : spip::DetectionPolarimetry("DetectionPolarimetryCUDA")
{
  stream = _stream;
}

spip::DetectionPolarimetryCUDA::~DetectionPolarimetryCUDA ()
{
}

void inline spip::DetectionPolarimetryCUDA::cross_detect (float p_r, float p_i, float q_r, float q_i,
                                                         float * pp, float * qq, float * pq_r, float * pq_i)
{
  *pp = (p_r * p_r) + (p_i * p_i);
  *qq = (q_r * q_r) + (q_i * q_i);
  *pq_r = (p_r * q_r) + (p_i * q_i);
  *pq_i = (p_r * q_i) - (p_i * q_r);
}

void inline spip::DetectionPolarimetryCUDA::stokes_detect (float p_r, float p_i, float q_r, float q_i,
                                                         float * s0, float * s1, float * s2, float * s3)
{
  const float pp = (p_r * p_r) + (p_i * p_i);
  const float qq = (q_r * q_r) + (q_i * q_i);
  *s0 = pp + qq;
  *s1  = pp - qq;
  *s2 = 2 * ((p_r * q_r) + (p_i * q_i));
  *s3 = 2 * ((p_r * q_i) + (p_i * q_r));
}


void spip::DetectionPolarimetryCUDA::transform_SFPT_to_SFPT ()
{
  if (verbose)
    cerr << "spip::DetectionPolarimetryCUDA::transform_SFPT_to_SFPT" << endl;
  throw invalid_argument ("spip::DetectionPolarimetryCUDA::transform_SFPT_to_SFPT not implemented (yet)");
}


void spip::DetectionPolarimetryCUDA::transform_TSPF_to_TSPF ()
{
  if (verbose)
    cerr << "spip::DetectionPolarimetryCUDA::transform_TSPF_to_TSPF()" << endl;
  throw invalid_argument ("spip::DetectionPolarimetryCUDA::transform_TSPF_to_TSPF not implemented (yet)");
}

void spip::DetectionPolarimetryCUDA::transform_TSPFB_to_TSPFB ()
{
  if (verbose)
    cerr << "spip::DetectionPolarimetryCUDA::transform_TSPFB_to_TSPFB()" << endl;
  throw invalid_argument ("spip::DetectionPolarimetryCUDA::transform_TSPFB_to_TSPFB not implemented (yet)");
}


void spip::DetectionPolarimetryCUDA::transform_TFPS_to_TFPS ()
{
  if (verbose)
    cerr << "spip::DetectionPolarimetryCUDA::transform_TFPS_to_TFPS" << endl;
  throw invalid_argument ("spip::DetectionPolarimetryCUDA::transform_TFPS_to_TFPS not implemented (yet)");
}
