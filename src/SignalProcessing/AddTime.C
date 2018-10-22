/***************************************************************************
 *
 *   Copyright (C) 2018 Andrew Jameson
 *   Licensed under the Academic Free License version 2.1
 *
 ***************************************************************************/

#include "spip/Error.h"
#include "spip/AddTime.h"

#include <stdexcept>
#include <cmath>

using namespace std;

spip::AddTime::AddTime (const char * name) : spip::Append (name)
{
  state = Signal::Intensity;
}

spip::AddTime::~AddTime ()
{
}

void spip::AddTime::set_output_state (spip::Signal::State _state)
{
  if ((state == spip::Signal::Intensity) || (state == spip::Signal::PPQQ))
    spip::Append::set_output_state (_state);
  else
    throw invalid_argument ("AddTime::set_output_state bad state");
}

//! intial configuration at the start of the data stream
void spip::AddTime::configure (spip::Ordering output_order)
{
  if (inputs.size() < 2)
    throw Error (InvalidState, "spip::AddTime::configure", "inputs not configured");

  ndat    = inputs[0]->get_ndat ();
  npol    = inputs[0]->get_npol ();
  nbit    = inputs[0]->get_nbit ();
  ndim    = inputs[0]->get_ndim ();
  nsignal = inputs[0]->get_nsignal ();
  nbin    = inputs[0]->get_nbin ();
  nchan   = inputs[0]->get_nchan ();

  //float nchan_sum = nchan;
  for (unsigned i=1; i<inputs.size(); i++)
  {
    if (ndat != inputs[i]->get_ndat())
      throw Error (InvalidState, "spip::AddTime::configure", "ndat did not match");
    if (npol != inputs[i]->get_npol())
      throw Error (InvalidState, "spip::AddTime::configure", "npol did not match");
    if (nbit != inputs[i]->get_nbit())
      throw Error (InvalidState, "spip::AddTime::configure", "nbit did not match");
    if (ndim != inputs[i]->get_ndim())
      throw Error (InvalidState, "spip::AddTime::configure", "ndim did not match");
    if (nsignal != inputs[i]->get_nsignal())
      throw Error (InvalidState, "spip::AddTime::configure", "ndat did not match");
    if (nbin != inputs[i]->get_nbin())
      throw Error (InvalidState, "spip::AddTime::configure", "nbin did not match");
    if (nchan != inputs[i]->get_nchan())
      throw Error (InvalidState, "spip::AddTime::configure", "nchan did not match");
  }

  bool order_same = true;
  for (unsigned i=0; i<inputs.size(); i++)
  {
    if (verbose)
      cerr << "spip::AddTime::configure inputs[i]->get_order()=" << inputs[i]->get_order() << " output_order=" << output_order << endl;
    if (inputs[i]->get_order() != output_order)
      order_same = false;
  }

  if (!order_same)
    throw invalid_argument ("AddTime::configure invalid ordering, inputs order different to output order");

  bool valid_combine = false;
  valid_combine |= (order_same && output_order == spip::Ordering::SFPT);
  valid_combine |= (order_same && output_order == spip::Ordering::TFPS);
  valid_combine |= (order_same && output_order == spip::Ordering::TSPF);
  valid_combine |= (order_same && output_order == spip::Ordering::TSPFB);

  if (!valid_combine)
    throw invalid_argument ("AddTime::configure invalid ordering");

  if (verbose)
    cerr << "spip::AddTime::configure ndat=" << ndat 
         << " ndim=" << ndim << " nbit=" << nbit << " npol=" << npol 
         << " nsignal=" << nsignal << endl;
  
  // determine the changed parameters

  // copy input header to output
  output->clone_header (inputs[0]->get_header());

  // output will read the newly cloned header parameters
  output->read_header ();

  output->set_order (output_order);

  // update the output header parameters with the new details
  output->write_header ();
  
  // resize the output container
  prepare_output();
}
