/***************************************************************************
 *
 *   Copyright (C) 2016 Andrew Jameson
 *   Licensed under the Academic Free License version 2.1
 *
 ***************************************************************************/

#include "spip/UnpackFloat.h"

#include <stdexcept>
#include <cmath>

using namespace std;

spip::UnpackFloat::UnpackFloat () : Transformation<Container,Container>("UnpackFloat", outofplace)
{
  offset = 0.0;
  scale = 1;
  endianness = spip::Little;
  encoding  = spip::TwosComplement;
  output_sideband = Signal::Sideband::None;
}

spip::UnpackFloat::~UnpackFloat ()
{
}

//! intial configuration at the start of the data stream
void spip::UnpackFloat::configure (spip::Ordering output_order)
{
  ndat  = input->get_ndat ();
  nchan = input->get_nchan ();
  npol  = input->get_npol ();
  nbit  = input->get_nbit ();
  ndim  = input->get_ndim ();
  nsignal = input->get_nsignal ();
  endianness = input->get_endianness();
  encoding = input->get_encoding ();
  sideband = input->get_sideband();

  if (verbose)
  {
    if (endianness == spip::Big)
      cerr << "spip::UnpackFloat::configure input is Big Endian" << endl;
    else
      cerr << "spip::UnpackFloat::configure input is Little Endian" << endl;
    if (encoding == spip::TwosComplement)
      cerr << "spip::UnpackFloat::configure input is Twos Complement" << endl;
    else
      cerr << "spip::UnpackFloat::configure input is Offset Binary" << endl;
    cerr << "spip::UnpackFloat::configure: ndat=" << ndat << endl;
  }

  if (ndim != 2)
    throw invalid_argument ("UnpackFloat::configure only ndim==2 supported");

  if (nbit != 8 && nbit != 16 && nbit != 32)
    throw invalid_argument ("UnpackFloat::configure nbit must be 8, 16 or 32");

  bool valid_transform = false;
  if (input->get_order() == spip::Ordering::SFPT && output_order == spip::Ordering::SFPT)
    valid_transform = true;
  if (input->get_order() == spip::Ordering::Custom && output_order == spip::Ordering::SFPT)
    valid_transform = true;
  if (!valid_transform)
    throw invalid_argument ("UnpackFloat::configure invalid ordering, must be SFPT->SFPT");

  // copy input header to output
  output->clone_header (input->get_header());

  // output will read the newly cloned header parameters
  output->read_header ();

  // update the parameters that this transformation will affect
  output->set_nbit (32);
  //output->set_instrument ("DSPSR");
  output->set_endianness (spip::Little);
  output->set_encoding (spip::TwosComplement);
  output->set_order (spip::Ordering::SFPT);

  // unpackers may change this in their consuctors
  if (output_sideband != Signal::Sideband::None)
  {
    output->set_sideband (output_sideband);
  }

  // update the output header parameters with the new details
  output->write_header ();
  
  // resize the output container
  prepare_output();
}

void spip::UnpackFloat::set_scale (float _scale)
{
  scale = _scale;
}

void spip::UnpackFloat::prepare ()
{
  ndat  = input->get_ndat ();
  if (verbose)
    cerr << "spip::UnpackFloat::prepare ndat=" << ndat << endl;
}

//! simply copy input buffer to output buffer
void spip::UnpackFloat::transformation ()
{
  if (verbose)
    cerr << "spip::UnpackFloat::transformation()" << endl;

  // ensure output is appropriately sized
  prepare_output ();

  // apply data transformation
  transform_custom_to_SFPT ();
}

void spip::UnpackFloat::prepare_output ()
{
  output->set_ndat (ndat);
  output->resize();
}

