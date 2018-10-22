/***************************************************************************
 *
 *   Copyright (C) 2016 Andrew Jameson
 *   Licensed under the Academic Free License version 2.1
 *
 ***************************************************************************/

#include "spip/Append.h"

#include <stdexcept>
#include <cmath>

using namespace std;

spip::Append::Append (const char * name) : Combination<Container,Container>(name)
{
  state = Signal::Intensity;
}

spip::Append::~Append ()
{
}

void spip::Append::set_output_state (spip::Signal::State _state)
{
  state = _state;
}

void spip::Append::prepare ()
{
  ndat = inputs[0]->get_ndat ();
  if (verbose)
    cerr << "spip::Append::prepare ndat=" << ndat << endl;
}

//! simply copy input buffer to output buffer
void spip::Append::combination ()
{
  if (verbose)
    cerr << "spip::Append::combination()" << endl;

  // ensure output is appropriately sized
  prepare_output ();

  bool order_same = true;
  for (unsigned i=0; i<inputs.size(); i++)
  {
    if (inputs[i]->get_order() != output->get_order())
      order_same = false;
  }
   
  if (!order_same)
    throw invalid_argument ("Append::combination invalid ordering of inputs and output");
  else if (output->get_order() == spip::Ordering::TSPF)
    combine_TSPF_to_TSPF();
  else if (output->get_order() == spip::Ordering::TSPFB) 
    combine_TSPFB_to_TSPFB();
  else
    throw invalid_argument ("Append::combination invalid ordering of inputs and output");
}

void spip::Append::prepare_output ()
{
  output->set_ndat (ndat);
  output->resize();
}

