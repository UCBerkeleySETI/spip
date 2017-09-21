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
  offset = 0;
  scale = 1;
}

spip::UnpackFloat::~UnpackFloat ()
{
}

//! intial configuration at the start of the data stream
void spip::UnpackFloat::configure ()
{
  ndat  = input->get_ndat ();
  nchan = input->get_nchan ();
  npol  = input->get_npol ();
  nbit  = input->get_nbit ();
  ndim  = input->get_ndim ();
  nsignal = input->get_nsignal ();

  if (verbose)
    cerr << "spip::UnpackFloat::configure: ndat=" << ndat << endl;

  if (ndim != 2)
    throw invalid_argument ("UnpackFloat::configure only ndim==2 supported");

  if (nbit != 8 && nbit != 16 && nbit != 32)
    throw invalid_argument ("UnpackFloat::configure nbit must be 8, 16 or 32");

  if (input->get_order() != spip::Ordering::SFPT)
    throw invalid_argument ("UnpackFloat::configure input order must be SFPT");

  // copy input header to output
  output->clone_header (input->get_header());

  // output will read the newly cloned header parameters
  output->read_header ();

  // update the parameters that this transformation will affect
  output->set_nbit (32);
  output->set_order (spip::Ordering::SFPT);

  // update the output header parameters with the new details
  output->write_header ();
  
  // resize the output container
  prepare_output();
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
  transform ();
}

void spip::UnpackFloat::prepare_output ()
{
  output->set_ndat (ndat);
  output->resize();
}
