/***************************************************************************
 *
 *   Copyright (C) 2018 Andrew Jameson
 *   Licensed under the Academic Free License version 2.1
 *
 ***************************************************************************/

#include "spip/Error.h"
#include "spip/IntegrationBinned.h"

#include <stdexcept>
#include <cmath>

using namespace std;

spip::IntegrationBinned::IntegrationBinned () : Transformation<Container,Container>("IntegrationBinned", outofplace)
{
  state = Signal::Intensity;

  dat_dec = 1;
  pol_dec = 1;
  chan_dec = 1;
  signal_dec = 1;

  buffer = NULL;
  buffer_idat = 0;
}

spip::IntegrationBinned::~IntegrationBinned ()
{
}

void spip::IntegrationBinned::set_decimation(unsigned _dat_dec, unsigned _pol_dec, unsigned _chan_dec, unsigned _signal_dec)
{
  dat_dec = _dat_dec;
  pol_dec = _pol_dec;
  chan_dec = _chan_dec;
  signal_dec = _signal_dec;
}

//! intial configuration at the start of the data stream
void spip::IntegrationBinned::configure (spip::Ordering output_order)
{
  ndat  = input->get_ndat ();
  nchan = input->get_nchan ();
  npol  = input->get_npol ();
  nbit  = input->get_nbit ();
  ndim  = input->get_ndim ();
  nsignal = input->get_nsignal ();
  tsamp = input->get_tsamp();
  nbin = 2;

  spip::AsciiHeader header = input->get_header();

  if (header.get ("CAL_SIGNAL", "%d", &cal_signal) != 1)
    throw Error (InvalidState, "IntegrationBinned::configure", "CAL_SIGNAL not present in header");
  if (header.get ("CAL_FREQ", "%lf", &cal_freq) != 1)
    throw Error (InvalidState, "IntegrationBinned::configure", "CAL_FREQ not present in header");
  if (header.get ("CAL_PHASE", "%lf", &cal_phase) != 1)
    throw Error (InvalidState, "IntegrationBinned::configure", "CAL_PHASE not present in header");
  if (header.get ("CAL_DUTY_CYCLE", "%lf", &cal_duty_cycle) != 1)
    throw Error (InvalidState, "IntegrationBinned::configure", "CAL_DUTY_CYCLE not present in header");
  char tmp_buf[128];
  if (header.get ("CAL_EPOCH", "%s", tmp_buf) != 1)
    throw Error (InvalidState, "IntegrationBinned::configure", "CAL_EPOCH not present in header");

  cal_epoch = new spip::Time(tmp_buf);

  // compute the difference between the UTC_START and CAL_EPOCH in seconds 
  cal_epoch_delta = int64_t(input->get_utc_start()->get_time()) - int64_t(cal_epoch->get_time());

  if (ndim != 1)
    throw invalid_argument ("IntegrationBinned::configure only ndim==1 supported");

  if (nbit != 32)
    throw invalid_argument ("IntegrationBinned::configure nbit must be 32");

  if (nchan % chan_dec != 0)
    throw invalid_argument ("IntegrationBinned::configure chan_dec must be a factor of nchan");

  if (pol_dec != 1 && npol != pol_dec)
    throw invalid_argument ("IntegrationBinned::configure pol_dec must be equal to npol");

  if (signal_dec != 1 && nsignal != signal_dec)
    throw invalid_argument ("IntegrationBinned::configure signal_dec must be equal to nsignal");

  bool valid_transform = false;
  if (input->get_order() == spip::Ordering::TSPF && output_order == spip::Ordering::TSPFB)
    valid_transform = true;
  if (!valid_transform)
    throw invalid_argument ("IntegrationBinned::configure invalid ordering, must be TSPF->TSPFB");

  // copy input header to output
  output->clone_header (input->get_header());

  // output will read the newly cloned header parameters
  output->read_header ();

  // update the parameters that this transformation will affect
  output->set_npol (npol / pol_dec);
  output->set_nchan (nchan / chan_dec);
  output->set_nsignal (nsignal / signal_dec);
  output->set_tsamp (tsamp * dat_dec);
  output->set_nbin (nbin);
  output->set_order (output_order);

  // update the output header parameters with the new details
  output->write_header ();
  
  // resize the output container
  prepare_output();

  // resize the folding plan 
  prepare_binplan();

  // now prepare the buffer
  if (verbose)
    cerr << "spip::IntegrationBinned::configure preparing buffer" << endl;
  buffer->clone_header(output->get_header());
  buffer->read_header();
  buffer->write_header();
  buffer->set_ndat (1);
  buffer->set_order (output_order);
  buffer->resize();
  buffer->zero();
  if (verbose)
    cerr << "spip::IntegrationBinned::configure completed" << endl;
}

void spip::IntegrationBinned::prepare ()
{
  ndat  = input->get_ndat ();
  if (verbose)
    cerr << "spip::IntegrationBinned::prepare ndat=" << ndat << endl;
}

//! simply copy input buffer to output buffer
void spip::IntegrationBinned::transformation ()
{
  if (verbose)
    cerr << "spip::IntegrationBinned::transformation()" << endl;

  // ensure output is appropriately sized
  prepare_output ();

  prepare_binplan ();

  // apply data transformation
  if (input->get_order() == spip::Ordering::TSPF && output->get_order() == spip::Ordering::TSPFB)
    transform_TSPF_to_TSPFB();
  else
    throw invalid_argument ("spip::IntegrationBinned::transformation unrecognized transformation state");
}

void spip::IntegrationBinned::prepare_output ()
{
  uint64_t output_ndat = (buffer_idat + ndat) / dat_dec;
  if (verbose)
    cerr << "spip::IntegrationBinned::prepare_output buffer_idat=" << buffer_idat 
         << " ndat=" << ndat << " dat_dec=" << dat_dec 
         << " output_ndat=" << output_ndat << endl;
  output->set_ndat (output_ndat);
  output->resize();
}

void spip::IntegrationBinned::prepare_binplan ()
{
  // for each time sample, determine the bin number 

  // resize the bin plan to be large enough
  binplan.resize (ndat);

  uint64_t end_idat = start_idat + ndat;

  unsigned bin;
  for (uint64_t idat=start_idat; idat < end_idat; idat++)
  {
    // convert the idat to an offset time from the utc_start [seconds]
    double idat_offset_utc_start_beg = (tsamp * idat) / 1000000;
    double idat_offset_utc_start_end = (tsamp * (idat+1)) / 1000000;

    // convert to offset from cal epoch [seconds]
    double idat_offset_cal_epoch_beg = idat_offset_utc_start_beg + (cal_epoch_delta);
    double idat_offset_cal_epoch_end = idat_offset_utc_start_end + (cal_epoch_delta);

    // convert to phase of the cal
    double phi_beg = (fmod (idat_offset_cal_epoch_beg, cal_period) / cal_period) - cal_phase;
    double phi_end = (fmod (idat_offset_cal_epoch_end, cal_period) / cal_period) - cal_phase;

    // if the starting phase is greater than the duty cycle
    if ((phi_beg > 0) && (phi_end < cal_duty_cycle))
      bin = 1;
    else if ((phi_beg > cal_duty_cycle) && (phi_end < 1))
      bin = 0;
    else
      bin = -1;

    binplan[idat-start_idat] = bin;
  }

  // for the next iteration
  start_idat = end_idat;
}
