/***************************************************************************
 *
 *   Copyright (C) 2016 Andrew Jameson
 *   Licensed under the Academic Free License version 2.1
 *
 ***************************************************************************/

#include "spip/AdaptiveFilter.h"

#include <stdexcept>
#include <cmath>

using namespace std;

spip::AdaptiveFilter::AdaptiveFilter () : Transformation<Container,Container>("AdaptiveFilter", outofplace)
{
  // default, should be configured [TODO]
  filter_update_time = 1000;
  epsilon = 1e-1;
  gains = NULL;
}

spip::AdaptiveFilter::~AdaptiveFilter ()
{
}

//! configure parameters at the start of a data stream
void spip::AdaptiveFilter::configure (spip::Ordering output_order)
{
  // this transformation requires the following parameters
  ndat  = input->get_ndat ();
  nbit = input->get_nbit ();
  ndim = input->get_ndim ();
  nchan = input->get_nchan ();
  npol  = input->get_npol ();
  nsignal = input->get_nsignal ();
  tsamp = input->get_tsamp();

  if (ndim != 2)
    throw invalid_argument ("AdaptiveFilter::configure input ndim != 2");

  if (nbit != 32)
    throw invalid_argument ("AdaptiveFilter::configure input nbit != 32");

  bool valid_transform = false;
  if (input->get_order() == spip::Ordering::TSPF && output_order == spip::Ordering::TSPF)
    valid_transform = true;
  if (input->get_order() == spip::Ordering::SFPT && output_order == spip::Ordering::SFPT)
    valid_transform = true;
  if (!valid_transform)
    throw invalid_argument ("AdaptiveFilter::configure invalid ordering, must be TSPF->TSPF, SFPT->SFPT");

  if (!gains)
    throw invalid_argument ("AdaptiveFilter::configure gains has not been allocated");

  // copy input header to output
  output->clone_header (input->get_header());

  // output will read the newly cloned header parameters
  output->read_header ();

  // adjust the required parameters
  output->set_order (input->get_order());

  // update the output header parameters with the new details
  output->write_header ();

  // ensure the output is appropriately sized
  prepare_output();

  // also prepare the gains
  if (verbose)
    cerr << "spip::AdaptiveFilter::configure gains->clone_header()" << endl;
  gains->clone_header (input->get_header());
  gains->read_header ();

  // update the parameters that this transformation will affect
  gains->set_nbit (32);
  gains->set_ndim (2);
  gains->set_order (spip::Ordering::TSPF);

  gains->write_header();
  gains->set_ndat (1);

  // allocate memory of the gains
  gains->resize();
  gains->zero();
}

//! prepare prior to each transformation call
void spip::AdaptiveFilter::prepare ()
{
  ndat = input->get_ndat();
}

//! ensure meta-data is correct in output
void spip::AdaptiveFilter::prepare_output ()
{
  if (verbose)
    cerr << "spip::AdaptiveFilter::prepare_output()" << endl;

  // update the output parameters that may change from block to block
  if (verbose)
    cerr << "spip::AdaptiveFilter::prepare_output output->set_ndat(" << ndat << ")" << endl;
  output->set_ndat (ndat);

  // resize output based on configuration
  if (verbose)
    cerr << "spip::AdaptiveFilter::prepare_output output->resize()" << endl;
  output->resize();

  if (verbose)
    cerr << "spip::AdaptiveFilter::prepare_output finished" << endl;
}

//! simply copy input buffer to output buffer
void spip::AdaptiveFilter::transformation ()
{
  if (verbose)
    cerr << "spip::AdaptiveFilter::transformation()" << endl;

  // ensure the output is appropriately sized
  prepare_output();

  if (ndat == 0)
  {
    cerr << "spip::AdaptiveFilter::transformation ndat==0, ignoring" << endl;
    return;
  }

  // apply data transformation
  if ((input->get_order() == TSPF) && (output->get_order() == TSPF))
  {
    if (verbose)
      cerr << "spip::AdaptiveFilter::transformation transform_TSPF()" << endl;
    transform_TSPF ();
  }
  else if ((input->get_order() == SFPT) && (output->get_order() == SFPT))
  {
    if (verbose)
      cerr << "spip::AdaptiveFilter::transformation transform_SFPT()" << endl;
    transform_SFPT ();
  }
  else
  {
    throw runtime_error ("AdaptiveFilter::transform unsupported input to output conversion");
  }
}

