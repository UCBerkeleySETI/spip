/***************************************************************************
 *
 *   Copyright (C) 2016 Andrew Jameson
 *   Licensed under the Academic Free License version 2.1
 *
 ***************************************************************************/

#include "spip/Add.h"

#include <stdexcept>
#include <cmath>

using namespace std;

spip::Add::Add (const char * name) : Combination<Container,Container>(name)
{
  state = Signal::Intensity;
}

spip::Add::~Add ()
{
}

void spip::Add::set_output_state (spip::Signal::State _state)
{
  state = _state;
}

void spip::Add::prepare ()
{
  ndat = inputs[0]->get_ndat ();
  if (verbose)
    cerr << "spip::Add::prepare ndat=" << ndat << endl;
}

//! simply copy input buffer to output buffer
void spip::Add::combination ()
{
  if (verbose)
    cerr << "spip::Add::combination()" << endl;

  // ensure output is appropriately sized
  prepare_output ();

  bool order_same = true;
  for (unsigned i=0; i<inputs.size(); i++)
  {
    if (inputs[i]->get_order() != output->get_order())
      order_same = false;
  }
   
  if (!order_same)
    throw invalid_argument ("Add::combination invalid ordering of inputs and output");
  else if (output->get_order() == spip::Ordering::TSPF)
    combine_TSPF_to_TSPF();
  else if (output->get_order() == spip::Ordering::TSPFB) 
    combine_TSPFB_to_TSPFB();
  else
    throw invalid_argument ("Add::combination invalid ordering of inputs and output");
}

void spip::Add::prepare_output ()
{
  output->set_ndat (ndat);
  output->resize();
}

