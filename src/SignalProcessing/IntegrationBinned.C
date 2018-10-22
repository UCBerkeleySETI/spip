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

  binplan = NULL;
  start_idat = 0;
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

  if (!input->get_cal_signal())
    throw Error (InvalidState, "IntegrationBinned::configure", "CAL_SIGNAL required in input");

  double cal_freq = input->get_cal_freq();
  if (cal_freq == 0)
    throw Error (InvalidState, "IntegrationBinned::configure", "CAL_FREQ was set to 0"); 
  cal_period = 1.0f / cal_freq;

  cal_phase = input->get_cal_phase();
  cal_duty_cycle = input->get_cal_duty_cycle();
  cal_epoch = input->get_cal_epoch();

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

  // prepare the binplan
  if (verbose)
    cerr << "spip::IntegrationBinned::configure preparing binplan" << endl;

  binplan->clone_header(output->get_header());
  binplan->read_header();
  binplan->set_npol (1);
  binplan->set_nsignal (1);
  binplan->set_ndim (1);
  binplan->set_nchan (1);
  binplan->set_ndat (1);
  binplan->set_order (output_order);  // doesn't reall matter (only has ndat dimension)
  binplan->write_header();
  binplan->resize();
  binplan->zero();

  // resize the folding plan in the subclass
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

  start_idat = 0;
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

/*
void spip::IntegrationBinned::prepare_binplan ()
{
  // for each time sample, determine the bin number 

  // resize the bin plan to be large enough
  if (verbose)
    cerr << "spip::IntegrationBinned::prepare_binplan binplan.resize (" << ndat << ")" << endl;
  binplan.resize (ndat);

  uint64_t end_idat = start_idat + ndat;

  if (verbose)
    cerr << "spip::IntegrationBinned::prepare_binplan start_idat=" << start_idat << " end_idat=" << end_idat << endl;
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

    // bin 0 == OFF bin 1 == ON
    // if the starting phase is greater than the duty cycle
    if ((phi_beg > 0) && (phi_end < cal_duty_cycle))
      bin = 1;
    else if ((phi_beg > cal_duty_cycle) && (phi_end < 1))
      bin = 0;
    else if ((phi_beg <0 ) && (phi_end < 0))
      bin = 0;
    else
      bin = -1;

    binplan[idat-start_idat] = bin;

    //if (idat < 1000000)
    //  cerr << "idat_offset=" << idat_offset_utc_start_beg << "," << idat_offset_utc_start_end << " idat_offset_cal_epoch=" << idat_offset_cal_epoch_beg << "," << idat_offset_cal_epoch_end << " phi=" << phi_beg << "," << phi_end << " bin=" << bin << " bin_idx= " << idat-start_idat << endl;

  }

  // for the next iteration
  start_idat = end_idat;
}
*/
