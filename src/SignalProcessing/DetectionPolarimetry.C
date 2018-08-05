/***************************************************************************
 *
 *   Copyright (C) 2016 Andrew Jameson
 *   Licensed under the Academic Free License version 2.1
 *
 ***************************************************************************/

#include "spip/DetectionPolarimetry.h"

#include <stdexcept>
#include <cmath>

using namespace std;

spip::DetectionPolarimetry::DetectionPolarimetry (const char * name) : spip::Detection (name)
{
  state = spip::Signal::Stokes;
}

spip::DetectionPolarimetry::~DetectionPolarimetry ()
{
}

void spip::DetectionPolarimetry::set_output_state (spip::Signal::State _state)
{
  if ((state == spip::Signal::Coherence) || (state == spip::Signal::Stokes))
    spip::Detection::set_output_state(_state);
  else
    throw invalid_argument ("DetectionPolarimetry::set_output_state bad state");   
}

//! intial configuration at the start of the data stream
void spip::DetectionPolarimetry::configure (spip::Ordering output_order)
{
  ndat  = input->get_ndat ();
  nchan = input->get_nchan ();
  npol  = input->get_npol ();
  nbit  = input->get_nbit ();
  ndim  = input->get_ndim ();
  nsignal = input->get_nsignal ();
  nbin = input->get_nbin ();

  if (ndim != 2)
    throw invalid_argument ("DetectionPolarimetry::configure only ndim==2 supported");

  if (nbit != 32)
    throw invalid_argument ("DetectionPolarimetry::configure nbit must be 32");

  if (npol != 2)
    throw invalid_argument ("DetectionPolarimetry::configure only npol==2 supported");

  bool valid_transform = false;
  if (input->get_order() == spip::Ordering::SFPT && output_order == spip::Ordering::SFPT)
    valid_transform = true;
  if (input->get_order() == spip::Ordering::TFPS && output_order == spip::Ordering::TFPS)
    valid_transform = true;
  if (input->get_order() == spip::Ordering::TSPF && output_order == spip::Ordering::TSPF)
    valid_transform = true;
  if (input->get_order() == spip::Ordering::TSPFB && output_order == spip::Ordering::TSPFB)
    valid_transform = true;
  if (!valid_transform)
    throw invalid_argument ("DetectionPolarimetry::configure invalid ordering, must be SFPT->SFPT");

  // copy input header to output
  output->clone_header (input->get_header());

  // output will read the newly cloned header parameters
  output->read_header ();

  // update the parameters that this transformation will affect
  output->set_ndim (1);
  output->set_npol (4);
  output->set_order (output_order);

  // update the output header parameters with the new details
  output->write_header ();
  
  // resize the output container
  prepare_output();
}
