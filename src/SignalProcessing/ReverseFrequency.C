/***************************************************************************
 *
 *   Copyright (C) 2018 Andrew Jameson
 *   Licensed under the Academic Free License version 2.1
 *
 ***************************************************************************/

#include "spip/ReverseFrequency.h"

#include <stdexcept>
#include <cmath>

using namespace std;

spip::ReverseFrequency::ReverseFrequency () : Transformation<Container,Container>("ReverseFrequency", outofplace)
{
  reversal = false;
  output_sideband = Signal::Sideband::None;
}

spip::ReverseFrequency::~ReverseFrequency ()
{
}

void spip::ReverseFrequency::set_sideband (spip::Signal::Sideband sideband)
{
  output_sideband = sideband;
}

//! intial configuration at the start of the data stream
void spip::ReverseFrequency::configure (spip::Ordering output_order)
{
  ndat  = input->get_ndat ();
  nchan = input->get_nchan ();
  npol  = input->get_npol ();
  nbit  = input->get_nbit ();
  ndim  = input->get_ndim ();
  nsignal = input->get_nsignal ();

  if (ndim != 1)
    throw invalid_argument ("ReverseFrequency::configure only ndim==1 supported");
  
  // TODO not really required
  if (nbit != 32)
    throw invalid_argument ("ReverseFrequency::configure only 32-bit floating point input required");

  if (output_sideband == Signal::Sideband::None)
    throw invalid_argument ("ReverseFrequency::configure output sideband not set");

  bool valid_transform = false;
  if (input->get_order() == spip::Ordering::TSPF && output_order == spip::Ordering::TSPF)
    valid_transform = true;
  if (input->get_order() == spip::Ordering::TFPS && output_order == spip::Ordering::TFPS)
    valid_transform = true;
  if (!valid_transform)
    throw invalid_argument ("ReverseFrequency::configure invalid ordering, must be TSPF->TSPF or TPFS->TPFS");

  // copy input header to output
  output->clone_header (input->get_header());

  // output will read the newly cloned header parameters
  output->read_header ();

  // update the parameters that this transformation will affect
  output->set_sideband (output_sideband);

  if (input->get_sideband() != output_sideband)
  {
    cerr << "spip::ReverseFrequency::prepare input sideband=" << input->get_sideband() 
         << " output_sideband=" << output_sideband << endl;
    output->set_bandwidth (input->get_bandwidth() * -1);
    reversal = true;
  }
  
  output->set_order (output_order);

  // update the output header parameters with the new details
  output->write_header ();
  
  // resize the output container
  prepare_output();
}

void spip::ReverseFrequency::prepare ()
{
  ndat  = input->get_ndat ();
  if (verbose)
    cerr << "spip::ReverseFrequency::prepare ndat=" << ndat << endl;
}

//! simply copy input buffer to output buffer
void spip::ReverseFrequency::transformation ()
{
  if (verbose)
    cerr << "spip::ReverseFrequency::transformation()" << endl;

  // ensure output is appropriately sized
  prepare_output ();

  // apply data transformation
  if (input->get_order() == spip::Ordering::TSPF && output->get_order() == spip::Ordering::TSPF)
    transform_TSPF_to_TSPF();
  else if (input->get_order() == spip::Ordering::TFPS && output->get_order() == spip::Ordering::TFPS)
    transform_TFPS_to_TFPS();
  else
    throw invalid_argument ("spip::ReverseFrequency::transformation unrecognized transformation state");
}

void spip::ReverseFrequency::prepare_output ()
{
  output->set_ndat (ndat);
  output->resize();
}

