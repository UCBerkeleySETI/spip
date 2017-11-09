/***************************************************************************
 *
 *   Copyright (C) 2016 Andrew Jameson
 *   Licensed under the Academic Free License version 2.1
 *
 ***************************************************************************/

#include "spip/UnpackFloatRAM.h"

#include <iostream>
#include <cstring>
#include <stdexcept>
#include <cmath>

using namespace std;

spip::UnpackFloatRAM::UnpackFloatRAM ()
{
}

spip::UnpackFloatRAM::~UnpackFloatRAM ()
{
}

void spip::UnpackFloatRAM::configure ()
{
  spip::UnpackFloat::configure ();
}

void spip::UnpackFloatRAM::prepare ()
{
  spip::UnpackFloat::prepare ();
}

void spip::UnpackFloatRAM::transform_SFPT_to_SFPT ()
{
  if (verbose)
    cerr << "spip::UnpackFloatRAM::transform_SFPT_to_SFPT" << endl;

  float * out = (float *) output->get_buffer();
  if (nbit == 8)
  {
    int8_t * in = (int8_t *) input->get_buffer();
    unpack_sfpt_to_sfpt(in, out);
  }
  else if (nbit == 16)
  {
    int16_t * in = (int16_t *) input->get_buffer();
    unpack_sfpt_to_sfpt (in, out);
  }
  else
  {
    float * in = (float *) input->get_buffer();
    unpack_sfpt_to_sfpt (in, out);
  } 
}
