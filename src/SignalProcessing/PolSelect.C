/***************************************************************************
 *
 *   Copyright (C) 2018 Andrew Jameson
 *   Licensed under the Academic Free License version 2.1
 *
 ***************************************************************************/

#include "spip/PolSelect.h"

#include <stdexcept>
#include <cmath>

using namespace std;

spip::PolSelect::PolSelect () : Transformation<Container,Container>("PolSelect", outofplace)
{
  delta_npol = 0;
  out_npol = 0;
}

spip::PolSelect::~PolSelect ()
{
}

//! set the number of polarisations to retain
void spip::PolSelect::set_pol_reduction (unsigned delta)
{
  delta_npol = delta;
}

//! configure parameters at the start of a data stream
void spip::PolSelect::configure (spip::Ordering output_order)
{
  // this transformation requires the following parameters
  ndat  = input->get_ndat ();
  nbit = input->get_nbit ();
  ndim = input->get_ndim ();
  nchan = input->get_nchan ();
  npol  = input->get_npol ();
  nsignal = input->get_nsignal ();

  out_npol = npol - delta_npol;

  if (nbit != 32)
    throw invalid_argument ("PolSelect::configure input nbit != 32");

  bool valid_transform = false;
  if (input->get_order() == spip::Ordering::TSPF && output_order == spip::Ordering::TSPF)
    valid_transform = true;
  if (input->get_order() == spip::Ordering::SFPT && output_order == spip::Ordering::SFPT)
    valid_transform = true;
  if (!valid_transform)
    throw invalid_argument ("PolSelect::configure invalid ordering, must be TSPF->TSPF, SFPT->SFPT");

  // copy input header to output
  output->clone_header (input->get_header());

  // output will read the newly cloned header parameters
  output->read_header ();

  // adjust the required parameters
  output->set_order (input->get_order());

  // set the output number of polarisations
  output->set_npol (out_npol);

  // update the output header parameters with the new details
  output->write_header ();

  // ensure the output is appropriately sized
  prepare_output();
}

//! prepare prior to each transformation call
void spip::PolSelect::prepare ()
{
  ndat = input->get_ndat();
}

//! ensure meta-data is correct in output
void spip::PolSelect::prepare_output ()
{
  if (verbose)
    cerr << "spip::PolSelect::prepare_output()" << endl;

  // update the output parameters that may change from block to block
  if (verbose)
    cerr << "spip::PolSelect::prepare_output output->set_ndat(" << ndat << ")" << endl;
  output->set_ndat (ndat);

  // resize output based on configuration
  if (verbose)
    cerr << "spip::PolSelect::prepare_output output->resize()" << endl;
  output->resize();

  if (verbose)
    cerr << "spip::PolSelect::prepare_output finished" << endl;
}

//! simply copy input buffer to output buffer
void spip::PolSelect::transformation ()
{
  if (verbose)
    cerr << "spip::PolSelect::transformation()" << endl;

  // ensure the output is appropriately sized
  prepare_output();

  if (ndat == 0)
  {
    cerr << "spip::PolSelect::transformation ndat==0, ignoring" << endl;
    return;
  }

  // if no polarisations are to be removed
  if (delta_npol == 0)
  {
    bypass ();
  } 
  // TSPF transformation
  else if ((input->get_order() == TSPF) && (output->get_order() == TSPF))
  {
    if (verbose)
      cerr << "spip::PolSelect::transformation transform_TSPF()" << endl;
    transform_TSPF ();
  }
  // SFPT transformation
  else if ((input->get_order() == SFPT) && (output->get_order() == SFPT))
  {
    transform_SFPT ();
  }
  else
  {
    throw runtime_error ("PolSelect::transform unsupported input to output conversion");
  }
}
