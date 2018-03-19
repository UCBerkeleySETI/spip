/***************************************************************************
 *
 *   Copyright (C) 2016 Andrew Jameson
 *   Licensed under the Academic Free License version 2.1
 *
 ***************************************************************************/

#include "spip/IntegerDelay.h"

#include <stdexcept>

using namespace std;

spip::IntegerDelay::IntegerDelay () : Transformation<Container,Container>("IntegerDelay", outofplace)
{
  prev_delays = NULL;
  curr_delays = NULL;
  delta_delays = NULL;
  have_buffered_output = false;
}

spip::IntegerDelay::~IntegerDelay ()
{
  if (prev_delays)
    delete prev_delays;
    if (curr_delays)
  delete curr_delays;
  if (delta_delays)
    delete delta_delays;
  if (buffered)
    delete buffered;
}

void spip::IntegerDelay::configure (spip::Ordering output_order)
{
  // this transformation requires the following parameters
  ndat  = input->get_ndat ();
  nbit = input->get_nbit ();
  ndim = input->get_ndim ();
  nchan = input->get_nchan ();
  npol  = input->get_npol ();
  nsignal = input->get_nsignal ();

  if (nbit != 32)
    throw invalid_argument ("IntegerDelay::configure input nbit != 32");

  bool valid_transform = false;
  if (input->get_order() == spip::Ordering::SFPT && output_order == spip::Ordering::SFPT)
    valid_transform = true;

  if (!valid_transform)
    throw invalid_argument ("AdaptiveFilter::configure invalid ordering, must be TSPF->TSPF, SFPT->SFPT");

  if (!prev_delays)
    throw invalid_argument ("AdaptiveFilter::configure prev_delays has not been allocated");
  if (!curr_delays)
    throw invalid_argument ("AdaptiveFilter::configure curr_delays has not been allocated");
  if (!delta_delays)
    throw invalid_argument ("AdaptiveFilter::configure delta_delays has not been allocated");

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

  // prepare the buffered output also
  buffered->clone_header (intput->get_header());
  buffered->read_header();
  buffered->write_header();

  // prepare the delay arrays
  prev_delays->clone_header (input->get_header());
  prev_delays->read_header();
  prev_delays->set_ndim (1);
  prev_delays->set_npol(1);
  prev_delays->set_nbit(sizeof(unsigned));
  prev_delays->set_order (spip::Ordering::TSPF);
  prev_delays->write_header();
  
  curr_delays->clone_header (prev->delays);
  curr_delays->read_header();
  curr_delays->write_header();

  delta_delays->clone_header (prev->delays);
  delta_delays->read_header();
  delta_delays->write_header();

  prev_delays->set_ndat (1);
  curr_delays->set_ndat (1);
  delta_delays->set_ndat (1);

  prev_delays->resize();
  curr_delays->resize();
  delta_delays->resize();

  prev_delays->zero();
  curr_delays->zero();
  delta_delays->zero();
}

//! prepare prior to each transformation call
void spip::IntegerDelay::prepare (unsigned nsignal)
{
  ndat = input->get_ndat ();

  buffered->set_ndat (ndat);
  buffered->resize ();
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

void spip::IntegerDelay::set_delay (unsigned isig, unsigned delay)
{
  if (isig >= curr_delays->get_nsignal())
    throw invalid_argument ("IntegerDelay::set_delay isig > nsignal");

  unsigned * buffer = (unsigned *) curr_delays->get_buffer();
  buffer[isig] = delay;
}

void spip::IntegerDelay::compute_delta_delays ()
{
  unsigned * prev_buffer = (unsigned *) prev_delays->get_buffer();
  unsigned * curr_buffer = (unsigned *) curr_delays->get_buffer();
  int * delta = (int *) delta_delays->get_buffer();

  for (unsigned isig=0; isig<nsignal; isig++)
    delta[isig] = (int) curr_buffer[isig] - (int) prev_buffer[isig];
}

//! simply copy input buffer to output buffer
void spip::IntegerDelay::transformation ()
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

  // determine the difference in delays
  compute_delta_delays();

  if ((input->get_order() == SFPT) && (output->get_order() == SFPT))
  {
    if (verbose)
      cerr << "spip::AdaptiveFilter::transformation transform_SFPT()" << endl;
    transform_SFPT ();
  }
  else
  {
    throw runtime_error ("AdaptiveFilter::transform unsupported input to output conversion");
  }

  // double buffer the delays
  spip::Container * tmp = prev_delays;
  prev_delays = curr_delays;
  curr_delays = tmp;

  // double buffer the output
  if (have_buffered_output)
  {
    tmp = dynamic_cast<spip::ContainerRAM *>(output);
    output = buffered;
    buffered = tmp;
    output->set_ndat (input->get_ndat());
  }
  else
  {
    have_buffered_output = true;
    output->set_ndat (0);
  }
}

