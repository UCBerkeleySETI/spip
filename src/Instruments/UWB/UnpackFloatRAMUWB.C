/***************************************************************************
 *
 *   Copyright (C) 2018 Andrew Jameson
 *   Licensed under the Academic Free License version 2.1
 *
 ***************************************************************************/

#include "spip/UnpackFloatRAMUWB.h"

#include <iostream>
#include <cstring>
#include <stdexcept>
#include <cmath>

using namespace std;

spip::UnpackFloatRAMUWB::UnpackFloatRAMUWB ()
{
}

spip::UnpackFloatRAMUWB::~UnpackFloatRAMUWB ()
{
}

void spip::UnpackFloatRAMUWB::prepare ()
{
  spip::UnpackFloat::prepare ();
  if (endianness != Little)
    throw Error(InvalidState, "spip::UnpackFloatRAMUWB::prepare", "Expecting Little Endian data");
  if (encoding != OffsetBinary)
    throw Error(InvalidState, "spip::UnpackFloatRAMUWB::prepare", "Expecting Offset Binary data");
  if (ndim != 2)
    throw Error(InvalidState, "spip::UnpackFloatRAMUWB::prepare", "Expecting complex sampled input");

  if (nchan != 1)
    throw Error(InvalidState, "spip::UnpackFloatRAMUWB::prepare", "Expecting 1 input channel"); 

  if (nsignal != 1)
    throw Error(InvalidState, "spip::UnpackFloatRAMUWB::prepare", "Expecting 1 input signal"); 
}

void spip::UnpackFloatRAMUWB::transform_custom_to_SFPT ()
{
  if (verbose)
    cerr << "spip::UnpackFloatRAMUWB::transform_custom_to_SFPT" << endl;

  // in/out
  float * out  = (float *) output->get_buffer();
  int16_t * in = (int16_t *) input->get_buffer();

  // input strides
  const uint64_t ndat_per_block = 2048;
  const uint64_t nblock = ndat / ndat_per_block;

  // output strides
  const uint64_t block_stride = ndat_per_block * ndim;
  const uint64_t pol_stride = ndat * ndim;
  const uint64_t nval = ndat_per_block * ndim;
  
  // input data are in BlockPolTime format, 1 channel, 1 signal
  for (uint64_t iblock=0; iblock<nblock; iblock++)
  {
    const uint64_t block_offset = iblock * block_stride;
    for (unsigned ipol=0; ipol<npol; ipol++)
    {
      const uint64_t pol_offset = block_offset + (ipol * pol_stride);
      for (uint64_t ival=0; ival<nval; ival++)
      {
        out[pol_offset + ival] = (convert_twos(*in) + offset) * scale;
        in++;
      }
    }
  }
}
