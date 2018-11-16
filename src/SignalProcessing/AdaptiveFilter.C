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

spip::AdaptiveFilter::AdaptiveFilter (const string& dir) : 
  Transformation<Container,Container>("AdaptiveFilter", outofplace),
  output_dir (dir) 
{
  // default, should be configured [TODO]
  perform_filtering = false;
  filter_update_time = 1024;
  epsilon = 0.1;
  gains = NULL;
  dirty = NULL;
  cleaned = NULL;
  norms = NULL;
  ref_pol = 0;
}

spip::AdaptiveFilter::~AdaptiveFilter ()
{
}

//! set filtering parameters, and indicate if a reference polarisation is present
void spip::AdaptiveFilter::set_filtering (int pol)
{
  ref_pol = pol;
  if (ref_pol < 0)
    throw invalid_argument ("AdaptiveFilter::set_filtering invalid reference pol");
  perform_filtering = true;
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

  // Now allow any pol to be the reference polarisation
  //if (ref_pol != int(npol) - 1)
  //  throw Error (InvalidState, "AdaptiveFilter::configure", "ref_pol [%d] != npol-1 [%d]\n",
  //               ref_pol, npol - 1);
  out_npol = npol - 1;

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

  if (!dirty)
    throw invalid_argument ("AdaptiveFilter::configure dirty has not been allocated");

  if (!cleaned)
    throw invalid_argument ("AdaptiveFilter::configure cleaned has not been allocated");

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

  // also prepare the gains
  if (verbose)
    cerr << "spip::AdaptiveFilter::configure gains->clone_header()" << endl;
  gains->clone_header (input->get_header());
  gains->read_header ();

  // update the parameters that this transformation will affect
  gains->set_nbit (32);
  gains->set_ndim (2);
  gains->set_tsamp (tsamp * ndat);
  gains->set_npol (out_npol);
  gains->set_order (spip::Ordering::TSPF);

  gains->write_header();
  gains->set_ndat (1);

  // allocate memory of the gains
  gains->resize();
  gains->zero();

  // prepare the normalization container
  norms->clone_header (input->get_header());
  norms->read_header ();

  // update the parameters that this transformation will affect
  norms->set_nbit (32);
  norms->set_ndim (1);
  norms->set_npol (out_npol);
  norms->set_order (spip::Ordering::TSPF);

  norms->write_header();
  norms->set_ndat (1);

  // allocate memory and initialize to zero
  norms->resize();
  norms->zero();
  
  // prepare the dirty container
  dirty->clone_header (input->get_header());
  dirty->read_header();

  // update the parameters that this transformation will affect
  dirty->set_nbit (32);
  dirty->set_ndim (1);
  dirty->set_tsamp (tsamp * ndat);
  dirty->set_npol (out_npol);
  dirty->set_order (spip::Ordering::TSPF);

  dirty->write_header();
  dirty->set_ndat (1);

  // allocate memory and initialize to zero
  dirty->resize();
  dirty->zero();

  // prepare the cleaned container
  cleaned->clone_header (input->get_header());
  cleaned->read_header();

  // update the parameters that this transformation will affect 
  cleaned->set_nbit (32);
  cleaned->set_ndim (1);
  cleaned->set_tsamp (tsamp * ndat);
  cleaned->set_npol (out_npol);
  cleaned->set_tsamp (tsamp * ndat);
  cleaned->set_order (spip::Ordering::TSPF);

  cleaned->write_header();
  cleaned->set_ndat (1);

  // allocate memory of the gains
  cleaned->resize();
  cleaned->zero();
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
    gains->set_ndat (0);
    dirty->set_ndat (0);
    cleaned->set_ndat (0);
    return;
  }
  else
  {
    gains->set_ndat(1);
    dirty->set_ndat(1);
    cleaned->set_ndat(1);

    gains->resize();
    dirty->resize();
    cleaned->resize();
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
    if (perform_filtering)
      transform_SFPT ();
  }
  else
  {
    throw runtime_error ("AdaptiveFilter::transform unsupported input to output conversion");
  }
}

