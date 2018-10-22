/***************************************************************************
 *
 *   Copyright (C) 2018 Andrew Jameson
 *   Licensed under the Academic Free License version 2.1
 *
 ***************************************************************************/

#include "spip/DetectionSquareLaw.h"

#include <stdexcept>
#include <cmath>

using namespace std;

spip::DetectionSquareLaw::DetectionSquareLaw (const char * name) : spip::Detection (name)
{
  state = Signal::Intensity;
}

spip::DetectionSquareLaw::~DetectionSquareLaw ()
{
}

void spip::DetectionSquareLaw::set_output_state (spip::Signal::State _state)
{
  if ((state == spip::Signal::Intensity) || (state == spip::Signal::PPQQ))
    spip::Detection::set_output_state (_state);
  else
    throw invalid_argument ("DetectionSquareLaw::set_output_state bad state");
}

//! intial configuration at the start of the data stream
void spip::DetectionSquareLaw::configure (spip::Ordering output_order)
{
  ndat  = input->get_ndat ();
  nchan = input->get_nchan ();
  npol  = input->get_npol ();
  nbit  = input->get_nbit ();
  ndim  = input->get_ndim ();
  nsignal = input->get_nsignal ();
  nbin = input->get_nbin ();

  if (ndim != 2)
    throw Error (InvalidState, "DetectionSquareLaw::configure",
                 "input ndim was %d, expecting 2", ndim);

  if (nbit != 32)
    throw invalid_argument ("DetectionSquareLaw::configure nbit must be 32");

  // can only form PPQQ if two input polarisations
  if ((state == spip::Signal::PPQQ) && (npol == 1))
    throw invalid_argument ("DetectionSquareLaw::configure cannot perform PPQQ if npol == 1");
  
  bool valid_transform = false;
  if (input->get_order() == spip::Ordering::SFPT && output_order == spip::Ordering::SFPT)
    valid_transform = true;
  if (input->get_order() == spip::Ordering::SFPT && output_order == spip::Ordering::TSPF)
    valid_transform = true;
  if (input->get_order() == spip::Ordering::TFPS && output_order == spip::Ordering::TFPS)
    valid_transform = true;
  if (input->get_order() == spip::Ordering::TSPF && output_order == spip::Ordering::TSPF)
    valid_transform = true;
  if (input->get_order() == spip::Ordering::TSPFB && output_order == spip::Ordering::TSPFB)
    valid_transform = true;
  if (!valid_transform)
    throw invalid_argument ("DetectionSquareLaw::configure invalid ordering, must be SFPT->SFPT");

  if (verbose)
    cerr << "spip::DetectionSquareLaw::configure nchan=" << nchan << " npol=" << npol << " nsignal=" << nsignal << endl;

  // copy input header to output
  output->clone_header (input->get_header());

  // output will read the newly cloned header parameters
  output->read_header ();

  // update the parameters that this transformation will affect
  output->set_ndim (1);

  // determine the type of detection to be performed
  if (state == spip::Signal::Intensity)
    output->set_npol (1);
  else if (state == spip::Signal::PPQQ)
    output->set_npol (2);

  output->set_order (output_order);

  // update the output header parameters with the new details
  output->write_header ();
  
  // resize the output container
  prepare_output();
}
