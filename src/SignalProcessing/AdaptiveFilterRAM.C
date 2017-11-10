/***************************************************************************
 *
 *   Copyright (C) 2016 Andrew Jameson
 *   Licensed under the Academic Free License version 2.1
 *
 ***************************************************************************/

#include "spip/AdaptiveFilterRAM.h"

#include <stdexcept>
#include <cmath>

using namespace std;

spip::AdaptiveFilterRAM::AdaptiveFilterRAM ()
{
}

spip::AdaptiveFilterRAM::~AdaptiveFilterRAM ()
{
}

void spip::AdaptiveFilterRAM::set_input_rfi (Container * _input_rfi)
{
  input_rfi = dynamic_cast<spip::ContainerRAM *>(_input_rfi);
  if (!input_rfi)
    throw Error (InvalidState, "spip::AdaptiveFilterRAM::set_input_rfi", 
                 "RFI input was not castable to spip::ContainerRAM *");
}

// configure the pipeline prior to runtime
void spip::AdaptiveFilterRAM::configure ()
{
  if (!gains)
    gains = new spip::ContainerRAM();
  spip::AdaptiveFilter::configure ();
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
  float * in_ref = (float *) input_rfi->get_buffer();
  float * out = (float *) output->get_buffer();
  float * gains_buf = (float *) gains->get_buffer();

  uint64_t idx  = 0;

  //complex float F, cleaned, corr, sum;
  float f_real, f_imag, af_real, af_imag, corr_real, corr_imag;
  float sum_real, sum_imag, c_real, c_imag, g_real, g_imag;

  // loop over the dimensions of the input block
  for (unsigned isig=0; isig<nsignal; isig++)
  {
    for (unsigned ichan=0; ichan<nchan; ichan++)
    {
      for (unsigned ipol=0; ipol<npol; ipol++)
      {
        sum_real = 0;
        sum_imag = 0;

        g_real = gains_buf[2*(ipol * nchan + ichan) + 0];
        g_imag = gains_buf[2*(ipol * nchan + ichan) + 1];

        for (uint64_t idat=0; idat<ndat; idat++)
        {
          const float ast_real = in[2*idx + 0];
          const float ast_imag = in[2*idx + 1];
          const float ref_real = in_ref[2*idx + 0];
          const float ref_imag = in_ref[2*idx + 1];

          // compute complex conjugate f = [gain * conj(ref)]
          f_real = g_real * ref_real - g_imag * ref_imag;
          f_imag = g_real * ref_imag + g_imag * ref_real;

          // subtract from the astronomy signal [af = ast - f]
          af_real = ast_real - f_real;
          af_imag = ast_imag - f_imag;

          // write af to output
          out[2*idx + 0] = af_real;
          out[2*idx + 1] = af_imag;

          // compute correlation [corr = af * conj(ref)]
          corr_real = af_real * ref_real + af_imag * ref_imag;
          corr_imag = af_imag * ref_real - af_real * ref_imag;

          // appened to the suma [sum = sum + corr]
          sum_real += corr_real;
          sum_imag += corr_imag;

          // update the filter
          if ((idat+1) % filter_update_time == 0)
          {
            // normalise the sum [c = sum / nsamp]
            c_real = sum_real / filter_update_time;
            c_imag = sum_imag / filter_update_time;

            // update the filter values [g = g + (c*epsilon)]
            g_real = g_real + c_real * epsilon;
            g_imag = g_imag + c_imag * epsilon;

            // reset sum to 0
            sum_real = 0;
            sum_imag = 0;
          }
          idx ++;
        }

        // save the gain for this pol/chan
        gains_buf[2*(ipol * nchan + ichan) + 0] = g_real;
        gains_buf[2*(ipol * nchan + ichan) + 1] = g_imag;
      }
    }
  }
}
