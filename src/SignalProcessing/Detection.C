/***************************************************************************
 *
 *   Copyright (C) 2016 Andrew Jameson
 *   Licensed under the Academic Free License version 2.1
 *
 ***************************************************************************/

#include "spip/Detection.h"

#include <stdexcept>
#include <cmath>

using namespace std;

spip::Detection::Detection (const char * name) : Transformation<Container,Container>(name, outofplace)
{
  state = Signal::Intensity;
}

spip::Detection::~Detection ()
{
}

void spip::Detection::set_output_state (spip::Signal::State _state)
{
  state = _state;
}

void spip::Detection::prepare ()
{
  ndat  = input->get_ndat ();
  if (verbose)
    cerr << "spip::Detection::prepare ndat=" << ndat << endl;
}

//! simply copy input buffer to output buffer
void spip::Detection::transformation ()
{
  if (verbose)
    cerr << "spip::Detection::transformation()" << endl;

  // ensure output is appropriately sized
  prepare_output ();

  if ((input->get_order() == spip::Ordering::SFPT) && (output->get_order() == spip::Ordering::SFPT))
    transform_SFPT_to_SFPT ();
  else if ((input->get_order() == spip::Ordering::TFPS) && (output->get_order() == spip::Ordering::TFPS))
    transform_TFPS_to_TFPS ();
  else if ((input->get_order() == spip::Ordering::TSPF) && (output->get_order() == spip::Ordering::TSPF))
    transform_TSPF_to_TSPF();
  else if ((input->get_order() == spip::Ordering::TSPFB) && (output->get_order() == spip::Ordering::TSPFB))
    transform_TSPFB_to_TSPFB();
  else
    throw invalid_argument ("Detection::transformation invalid ordering, must be");
}

void spip::Detection::prepare_output ()
{
  output->set_ndat (ndat);
  output->resize();
}

