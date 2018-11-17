/***************************************************************************
 *
 *   Copyright (C) 2018 Andrew Jameson
 *   Licensed under the Academic Free License version 2.1
 *
 ***************************************************************************/

#include "spip/UnpackFloatRAMUWB.h"
#include "spip/Types.h"

#include <iostream>
#include <cstring>
#include <stdexcept>
#include <cmath>

using namespace std;

spip::UnpackFloatRAMUWB::UnpackFloatRAMUWB ()
{
  scale = 1.0;
  offset = 0.0;
  output_sideband = Signal::Sideband::Upper;
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

  // check if a change in sideband is required
  if (sideband != output_sideband)
  {
    re_scale = scale;
    im_scale = scale * -1;
  }
  else
  {
    re_scale = scale;
    im_scale = scale;
  }
}


void spip::UnpackFloatRAMUWB::transform_custom_to_SFPT ()
{
  if (verbose)
    cerr << "spip::UnpackFloatRAMUWB::transform_custom_to_SFPT offset="
         << offset << " scale=" << scale << endl;

  // in/out
  float * out  = (float *) output->get_buffer();
  int16_t * in = (int16_t *) input->get_buffer();

  // input strides
  const uint64_t ndat_per_block = 2048;
  const uint64_t nblock = ndat / ndat_per_block;

  // output strides
  const uint64_t block_stride = ndat_per_block * ndim;
  const uint64_t pol_stride = ndat * ndim;

  // input data are in BlockPolTime format, 1 channel, 1 signal
  for (uint64_t iblock=0; iblock<nblock; iblock++)
  {
    const uint64_t block_offset = iblock * block_stride;
    for (unsigned ipol=0; ipol<npol; ipol++)
    {
      const uint64_t pol_offset = block_offset + (ipol * pol_stride);
      for (uint64_t idat=0; idat<ndat_per_block; idat++)
      {
        // real
        out[pol_offset + (2*idat) + 0] = (convert_twos(*in) + offset) * re_scale;
        in++;

        // imag
        out[pol_offset + (2*idat) + 1] = (convert_twos(*in) + offset) * im_scale;
        in++;
      }
    }
  }
}
