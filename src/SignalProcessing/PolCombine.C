/***************************************************************************
 *
 *   Copyright (C) 2018 Andrew Jameson
 *   Licensed under the Academic Free License version 2.1
 *
 ***************************************************************************/

#include "spip/PolCombine.h"

#include <stdexcept>
#include <cmath>

using namespace std;

spip::PolCombine::PolCombine () : Combination<Container,Container>("PolCombine")
{
  total_npol = 0;
}

spip::PolCombine::~PolCombine ()
{
}

//! configure parameters at the start of a data stream
void spip::PolCombine::configure (spip::Ordering output_order)
{
  ndat = inputs[0]->get_ndat ();
  nbit = inputs[0]->get_nbit ();
  ndim = inputs[0]->get_ndim ();
  nchan = inputs[0]->get_nchan ();
  nsignal = inputs[0]->get_nsignal ();

  bool matching = true;
  total_npol = inputs[0]->get_npol();
  for (unsigned i=1; i<inputs.size(); i++)
  {
    if (ndat != inputs[i]->get_ndat())
      matching = false;
    if (nbit != inputs[i]->get_nbit())
      matching = false;
    if (ndim != inputs[i]->get_ndim())
      matching = false;
    if (nchan != inputs[i]->get_nchan())
      matching = false;
    if (nsignal != inputs[i]->get_nsignal())
      matching = false;
    if (inputs[0]->get_order() != inputs[i]->get_order())
      matching = false;
    total_npol += inputs[i]->get_npol();
  }

  if (!matching)
    throw invalid_argument ("PolCombine::configure inputs not matching");

  bool valid_combine = false;
  if (inputs[0]->get_order() == spip::Ordering::SFPT && output_order == spip::Ordering::SFPT)
    valid_combine = true;
  if (!valid_combine)
    throw invalid_argument ("PolCombine::configure invalid ordering, must be SFPT->SFPT");

  // copy input header to output
  output->clone_header (inputs[0]->get_header());

  // output will read the newly cloned header parameters
  output->read_header ();

  // adjust the required parameters
  output->set_order (inputs[0]->get_order());

  // set the output number of polarisations
  output->set_npol (total_npol);

  // update the output header parameters with the new details
  output->write_header ();

  cerr << "PolCombine::configure total_npol=" << total_npol << endl;
  //cerr << "output header=" << output->get_header().raw() << endl;

  // ensure the output is appropriately sized
  prepare_output();
}

//! prepare prior to each combination call
void spip::PolCombine::prepare ()
{
  ndat = inputs[0]->get_ndat();
}

//! ensure meta-data is correct in output
void spip::PolCombine::prepare_output ()
{
  if (verbose)
    cerr << "spip::PolCombine::prepare_output()" << endl;

  // update the output parameters that may change from block to block
  if (verbose)
    cerr << "spip::PolCombine::prepare_output output->set_ndat(" << ndat << ")" << endl;
  output->set_ndat (ndat);

  // resize output based on configuration
  if (verbose)
    cerr << "spip::PolCombine::prepare_output output->resize()" << endl;
  output->resize();

  if (verbose)
    cerr << "spip::PolCombine::prepare_output finished" << endl;
}

//! 
void spip::PolCombine::combination()
{
  if (verbose)
    cerr << "spip::PolCombine::combination()" << endl;

  // ensure the output is appropriately sized
  prepare_output();

  if (ndat == 0)
  {
    cerr << "spip::PolCombine::combination ndat==0, ignoring" << endl;
    return;
  }

  // apply data combination
  if ((inputs[0]->get_order() == SFPT) && (output->get_order() == SFPT))
  {
    if (verbose)
      cerr << "spip::PolCombine::combination combine_SFPT()" << endl;
    combine_SFPT ();
  }
  else
  {
    throw runtime_error ("PolCombine::combination unsupported input to output conversion");
  }
}
