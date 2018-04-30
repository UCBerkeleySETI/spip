/***************************************************************************
 *
 *   Copyright (C) 2018 Andrew Jameson
 *   Licensed under the Academic Free License version 2.1
 *
 ***************************************************************************/

#include "spip/SampleFoldCUDA.h"
#include "spip/ContainerCUDA.h"

#include <iostream>
#include <cstring>
#include <stdexcept>
#include <cmath>

using namespace std;

spip::SampleFoldCUDA::SampleFoldCUDA (cudaStream_t _stream)
{
  stream = _stream;
}

spip::SampleFoldCUDA::~SampleFoldCUDA ()
{
}

void spip::SampleFoldCUDA::configure (spip::Ordering output_order)
{
  if (!buffer)
    buffer = new spip::ContainerCUDADevice();
  spip::SampleFold::configure (output_order);
}

void spip::SampleFoldCUDA::transform_SFPT_to_TSPFB ()
{
  if (verbose)
    cerr << "spip::SampleFoldCUDA::transform_SFPT_to_TSPFB" << endl;
  throw Error (InvalidState, "spip::SampleFoldCUDA::transform_SFPT_to_TSPFB", "not implemented");
}

void spip::SampleFoldCUDA::transform_TSPF_to_TSPFB ()
{
  if (verbose)
    cerr << "spip::SampleFoldCUDA::transform_TSPF_to_TSPFB" << endl;
  throw Error (InvalidState, "spip::SampleFoldCUDA::transform_TFPS_to_TSPFB", "not implemented");
}
          
void spip::SampleFoldCUDA::transform_TFPS_to_TSPFB ()
{
  if (verbose)
    cerr << "spip::SampleFoldCUDA::transform_TFPS_to_TSPFB" << endl;
  throw Error (InvalidState, "spip::SampleFoldCUDA::transform_TFPS_to_TSPFB", "not implemented");
}
