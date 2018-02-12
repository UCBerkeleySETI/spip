/***************************************************************************
 *
 *   Copyright (C) 2016 Andrew Jameson
 *   Licensed under the Academic Free License version 2.1
 *
 ***************************************************************************/

#include "spip/AdaptiveFilterRAM.h"

#include <stdexcept>
#include <float.h>
#include <complex.h>
#include <cmath>

using namespace std;

spip::AdaptiveFilterRAM::AdaptiveFilterRAM ()
{
}

spip::AdaptiveFilterRAM::~AdaptiveFilterRAM ()
{
}

void spip::AdaptiveFilterRAM::set_input_ref (Container * _input_ref)
{
  input_ref = dynamic_cast<spip::ContainerRAM *>(_input_ref);
  if (!input_ref)
    throw Error (InvalidState, "spip::AdaptiveFilterRAM::set_input_ref", 
                 "RFI input was not castable to spip::ContainerRAM *");
}

// configure the pipeline prior to runtime
void spip::AdaptiveFilterRAM::configure (spip::Ordering output_order)
{
  if (!gains)
    gains = new spip::ContainerRAM();
  spip::AdaptiveFilter::configure (output_order);
}

//! no special action required
void spip::AdaptiveFilterRAM::prepare ()
{
  spip::AdaptiveFilter::prepare();
}

void spip::AdaptiveFilterRAM::transform_TSPF()
{
  if (verbose)
    cerr << "spip::AdaptiveFilterRAM::transform_TSPF ()" << endl;
}

void spip::AdaptiveFilterRAM::transform_SFPT()
{
  if (verbose)
    cerr << "spip::AdaptiveFilterRAM::transform_SFPT ()" << endl;

  // pointers to the buffers for in, rfi and out
  float * in = (float *) input->get_buffer();
  float * in_ref = (float *) input_ref->get_buffer();
  float * out = (float *) output->get_buffer();
  float * gains_buf = (float *) gains->get_buffer();

  uint64_t idat  = 0;

  float f_real, f_imag, af_real, af_imag, corr_real, corr_imag;
  float sum_real, sum_imag, c_real, c_imag, g_real, g_imag;

  if (verbose)
    cerr << "spip::AdaptiveFilterRAM::transform_SFPT ndat=" << ndat 
         << " nloops=" << ndat / filter_update_time << endl;

  // segregation into blocks of length filter_update_time
  uint64_t nblocks = ndat / filter_update_time;
  if (nblocks % filter_update_time != 0)
    nblocks++;

  // loop over the dimensions of the input block
  for (unsigned isig=0; isig<nsignal; isig++)
  {
    for (unsigned ichan=0; ichan<nchan; ichan++)
    {
      for (unsigned ipol=0; ipol<npol; ipol++)
      {
        g_real = gains_buf[2*(ipol * nchan + ichan) + 0];
        g_imag = gains_buf[2*(ipol * nchan + ichan) + 1];

        // we compute the filter over blocks
        for (uint64_t iblock=0; iblock<nblocks; iblock++)
        {
          // reset the sum for this block
          sum_real = 0;
          sum_imag = 0;

          // starting time sample for this block
          idat = iblock * filter_update_time;
          unsigned count = 0;
          for (uint64_t ival=0; ival<filter_update_time; ival++)
          {
            if (idat < ndat)
            {
              const float ast_real = in[2*idat + 0];
              const float ast_imag = in[2*idat + 1];

              const float ref_real = in_ref[2*idat + 0];
              const float ref_imag = in_ref[2*idat + 1];

              // compute f = [gain * ref]
              f_real = g_real * ref_real - g_imag * ref_imag;
              f_imag = g_real * ref_imag + g_imag * ref_real;

              // subtract from the astronomy signal [af = ast - f]
              af_real = ast_real - f_real;
              af_imag = ast_imag - f_imag;

              // compute correlation [corr = af * conj(ref)]
              corr_real = af_real * ref_real + af_imag * ref_imag;
              corr_imag = af_imag * ref_real - af_real * ref_imag;

              // appened to the suma [sum = sum + corr]
              sum_real += corr_real;
              sum_imag += corr_imag;
              count++;
            }
            idat++;
          }

          // update the filter, first normalise the sum [c = sum / nsamp]
          c_real = sum_real / filter_update_time;
          c_imag = sum_imag / filter_update_time;

          // update the filter values [g = g + (c*epsilon)]
          g_real = g_real + c_real * epsilon;
          g_imag = g_imag + c_imag * epsilon;

          idat = iblock * filter_update_time;
          for (uint64_t ival=0; ival<filter_update_time; ival++)
          {
            if (idat < ndat)
            {
              // re-read the astronomy signal
              const float ast_real = in[2*idat + 0];
              const float ast_imag = in[2*idat + 1];
              //const complex float ast = in[idat];

              // re-read the reference signal
              const float ref_real = in_ref[2*idat + 0];
              const float ref_imag = in_ref[2*idat + 1];
              //const complex ref = in_ref[idat];

              // compute f = [gain * ref]
              f_real = g_real * ref_real - g_imag * ref_imag;
              f_imag = g_real * ref_imag + g_imag * ref_real;
              //f = g * ref;

              // subtract from the astronomy signal [af = ast - f]
              af_real = ast_real - f_real;
              af_imag = ast_imag - f_imag;
              //af = ast - f;

              // write af to output
              out[2*idat + 0] = af_real;
              out[2*idat + 1] = af_imag;
              //out[2*idat] = af;

              //const complex ref = in_ref[idat];
              //f = g * ref;
              //af = ast - f;
              //out[idat] = af;
            }
            idat++;
          }
        }
        // save the gain for this pol/chan
        gains_buf[2*(ipol * nchan + ichan) + 0] = g_real;
        gains_buf[2*(ipol * nchan + ichan) + 1] = g_imag;
      }
    }
  }
}
