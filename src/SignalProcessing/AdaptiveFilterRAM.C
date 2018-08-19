/***************************************************************************
 *
 *   Copyright (C) 2016 Andrew Jameson
 *   Licensed under the Academic Free License version 2.1
 *
 ***************************************************************************/

#include "spip/AdaptiveFilterRAM.h"
#include "spip/Error.h"

#include <float.h>
#include <string.h>
#include <complex.h>
#include <cmath>

using namespace std;

spip::AdaptiveFilterRAM::AdaptiveFilterRAM (string dir) : AdaptiveFilter (dir)
{
  processed_first_block = false;
}

spip::AdaptiveFilterRAM::~AdaptiveFilterRAM ()
{
  if (gains)
    delete gains;
  gains = NULL;

  if (norms)
    delete norms;
  norms = NULL;
}

// configure the pipeline prior to runtime
void spip::AdaptiveFilterRAM::configure (spip::Ordering output_order)
{
  if (!gains)
    gains = new spip::ContainerFileWrite (output_dir);

  if (!norms)
    norms = new spip::ContainerRAM ();

  int64_t gains_size = nchan * out_npol * ndim * sizeof(float);
  gains_file_write = dynamic_cast<spip::ContainerFileWrite *>(gains);
  gains_file_write->set_file_length_bytes (gains_size);

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
  throw Error (InvalidState, "spip::AdaptiveFilterRAM::transform_TSPF", "not implemented");
}

void spip::AdaptiveFilterRAM::transform_SFPT()
{
  if (verbose)
    cerr << "spip::AdaptiveFilterRAM::transform_SFPT ()" << endl;

  // pointers to the buffers for in, rfi and out
  float * in = (float *) input->get_buffer();
  float * out = (float *) output->get_buffer();
  float * gains_buf = (float *) gains->get_buffer();
  float * norms_buf = (float *) norms->get_buffer();

  float re, im;
  float f_real, f_imag, af_real, af_imag, corr_real, corr_imag;
  float sum_real, sum_imag, c_real, c_imag, g_real, g_imag;
  float ast_real, ast_imag, ref_real, ref_imag;

  float ast_sum, ref_sum;
  float normalized_power, current_factor, normalized_factor;
  uint64_t idat, nval;

  // segregation into blocks of length filter_update_time
  uint64_t nblocks = ndat / filter_update_time;
  if (ndat% filter_update_time != 0)
    nblocks++;

  if (verbose)
    cerr << "spip::AdaptiveFilterRAM::transform_SFPT ndat=" << ndat
         << " nblocks=" << nblocks << " npol=" << npol << " out_npol=" << out_npol << endl;

  // strides through the data block
  const uint64_t pol_stride = ndat * ndim;
  const uint64_t in_chan_stride  = npol * pol_stride;
  const uint64_t out_chan_stride = out_npol * pol_stride;
  const uint64_t in_sig_stride  = nchan * in_chan_stride;
  const uint64_t out_sig_stride = nchan * out_chan_stride;

  // loop over the dimensions of the input block
  for (unsigned isig=0; isig<nsignal; isig++)
  {
    const uint64_t in_sig_offset  = isig * in_sig_stride;
    const uint64_t out_sig_offset = isig * out_sig_stride;
    for (unsigned ichan=0; ichan<nchan; ichan++)
    {
      const uint64_t in_chan_offset = in_sig_offset + ichan * in_chan_stride;
      const uint64_t out_chan_offset = out_sig_offset + ichan * out_chan_stride;

      for (unsigned ipol=0; ipol<out_npol; ipol++)
      {
        unsigned ast_pol = (int(ipol) < ref_pol) ? ipol : ipol + 1;
        const uint64_t in_sfp_offset = in_chan_offset + ast_pol * pol_stride;
        const uint64_t in_ref_offset = in_chan_offset + ref_pol * pol_stride;
        const uint64_t out_sfp_offset = out_chan_offset + ipol * pol_stride;

        // gains are stored in TSPF [ndat==1]
        const unsigned gain_ndim = 2;
        const uint64_t gain_offset = (isig * out_npol * nchan * gain_ndim) + (ipol * nchan * gain_ndim) + (ichan * gain_ndim);

        // read previous gain values
        g_real = gains_buf[gain_offset + 0];
        g_imag = gains_buf[gain_offset + 1];

        const uint64_t norms_offset = (isig * out_npol * nchan) + (ipol * nchan) + ichan;
        previous_factor = norms_buf[norms_offset];

        // offset pointers for this signal, channel and polarisation
        float * in_ptr  = in + in_sfp_offset;
        float * ref_ptr = in + in_ref_offset;
        float * out_ptr = out + out_sfp_offset;

        // we compute the filter over blocks of data
        for (uint64_t iblock=0; iblock<nblocks; iblock++)
        {
          // compute sum of ast and ref signals
          ast_sum = 0;
          ref_sum = 0;
          nval = 0;

          idat = (iblock * filter_update_time);

          for (uint64_t ival=0; ival<filter_update_time; ival++)
          {
            if (idat < ndat)
            {
              re = in_ptr[(2*idat)+0];
              im = in_ptr[(2*idat)+1];
              ast_sum += re*re + im*im;

              re = ref_ptr[(2*idat)+0];
              im = ref_ptr[(2*idat)+1];
              ref_sum += re*re + im*im;
              nval++;
            }
            idat++;
          }

          normalized_power = (ast_sum / nval) + (ref_sum/ nval);
          current_factor   = normalized_power;

          if (processed_first_block)
            normalized_factor = (0.999 * previous_factor) + (0.001 * current_factor);
          else
            normalized_factor = current_factor;
          previous_factor = normalized_factor;
          processed_first_block = true;

          // reset the sum for this block
          sum_real = 0;
          sum_imag = 0;

          // starting time sample for this block
          idat = (iblock * filter_update_time);

          // loop only over the actual number of values
          for (uint64_t ival=0; ival<nval; ival++)
          {
            const uint64_t re_idat = (2*idat) + 0;
            const uint64_t im_idat = (2*idat) + 1;

            ast_real = in_ptr[re_idat];
            ast_imag = in_ptr[im_idat];

            ref_real = ref_ptr[re_idat];
            ref_imag = ref_ptr[im_idat];

            // compute f = [gain * ref]
            f_real = g_real * ref_real - g_imag * ref_imag;
            f_imag = g_real * ref_imag + g_imag * ref_real;

            // subtract from the astronomy signal [af = ast - f]
            af_real = ast_real - f_real;
            af_imag = ast_imag - f_imag;

            // compute correlation [corr = af * conj(ref)]
            corr_real = (af_real * ref_real + af_imag * ref_imag) / normalized_factor;
            corr_imag = (af_imag * ref_real - af_real * ref_imag) / normalized_factor;

            // appened to the suma [sum = sum + corr]
            sum_real += corr_real;
            sum_imag += corr_imag;

            idat++;
          }

          // update the filter, first normalise the sum [c = sum / nsamp]
          c_real = sum_real / nval;
          c_imag = sum_imag / nval;

          // update the filter values [g = g + (c*epsilon)]
          g_real = g_real + (c_real * epsilon);
          g_imag = g_imag + (c_imag * epsilon);

          idat = iblock * filter_update_time;

          for (uint64_t ival=0; ival<nval; ival++)
          {
            const uint64_t re_idat = (2*idat) + 0;
            const uint64_t im_idat = (2*idat) + 1;

            // the astronomy signal
            ast_real = in_ptr[re_idat];
            ast_imag = in_ptr[im_idat];

            // the reference signal
            ref_real = ref_ptr[re_idat];
            ref_imag = ref_ptr[im_idat];

            // compute f = [gain * ref]
            f_real = g_real * ref_real - g_imag * ref_imag;
            f_imag = g_real * ref_imag + g_imag * ref_real;

            // subtract from the astronomy signal [af = ast - f]
            af_real = ast_real - f_real;
            af_imag = ast_imag - f_imag;

            // write af to output
            out_ptr[re_idat] = af_real;
            out_ptr[im_idat] = af_imag;
            idat++;
          }
        }

        // save the gain for this pol/chan
        gains_buf[gain_offset + 0] = g_real;
        gains_buf[gain_offset + 1] = g_imag;

        // save the normalization for this sig/pol/chan
        norms_buf[norms_offset] = current_factor;

      }
    }
  }
}

void spip::AdaptiveFilterRAM::write_gains ()
{
  // write the current values of the gains (for each polarisation and channel) to file
  gains_file_write->write();
}
