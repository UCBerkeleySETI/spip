/***************************************************************************
 *
 *   Copyright (C) 2016 Andrew Jameson
 *   Licensed under the Academic Free License version 2.1
 *
 ***************************************************************************/

#include "spip/AdaptiveFilterCUDA.h"
#include "spip/Error.h"

#include <iostream>
#include <cmath>
#include <cuComplex.h>
#include <cstdio>

using namespace std;

// compute a sum of a cuFloatComplex across a warp
__inline__ __device__
cuFloatComplex warpReduceSumFC(cuFloatComplex val) 
{
  for (int offset = warpSize/2; offset > 0; offset /= 2)
  {
    val.x += __shfl_down(val.x, offset);
    val.y += __shfl_down(val.y, offset);
  }
  return val;
}

// compute a sum of a float across a block
__inline__ __device__
cuFloatComplex blockReduceSumFC(cuFloatComplex val) 
{
  __shared__ cuFloatComplex shared[32]; // shared mem for 32 partial sums

  int lane = threadIdx.x % warpSize;
  int wid = threadIdx.x / warpSize;

  val = warpReduceSumFC(val);     // Each warp performs partial reduction

  if (lane==0) shared[wid] = val; // Write reduced value to shared memory

  __syncthreads();              // Wait for all partial reductions

  //read from shared memory only if that warp existed
  val = (threadIdx.x < blockDim.x / warpSize) ? shared[lane] : make_cuFloatComplex(0, 0);

  if (wid==0) val = warpReduceSumFC(val); //Final reduce within first warp

  return val;
}


// SFPT implementation of adaptive filter algorithm
__global__ void AdaptiveFilterKernel_SFPT (cuFloatComplex * in, cuFloatComplex * out, cuFloatComplex * gains, 
                                           float * dirty, float * cleaned, float * norms,
                                           uint64_t nloops, bool processed_first_block,
                                           uint64_t in_sig_stride, uint64_t in_chan_stride, 
                                           uint64_t out_sig_stride, uint64_t out_chan_stride,
                                           uint64_t pol_stride, unsigned ref_pol)
{
  // there is one complex gain per block
  __shared__ cuFloatComplex g;
  float previous_factor;
  cuFloatComplex cleaned_power_sum;
  
  cuFloatComplex pa, pr;

  // Block X = Signal
  // Block Y = Channel
  // Block Z = Polarisation
  // Gains are nominally stored in TSPF where T == 1, so SPF
  const unsigned isigchanpol = (blockIdx.x * gridDim.y * gridDim.z) + 
                               (blockIdx.z * gridDim.y) + 
                               blockIdx.y;

  // get the first thread to read the previously computed complex gain
  if (threadIdx.x == 0)
  {
    g = gains[isigchanpol];
    previous_factor = norms[isigchanpol];
  }

  // ensure all threads have the complex gain
  __syncthreads();

  // the reference polarisation could be (unfortunately) not the final ones
  unsigned ast_pol = (blockIdx.z < ref_pol) ? blockIdx.z : blockIdx.z + 1;

  // offsets for input, reference and output, stored in SFPT
  uint64_t idx = (blockIdx.x * in_sig_stride)  + (blockIdx.y * in_chan_stride)  + (ast_pol * pol_stride)    + threadIdx.x;
  uint64_t rdx = (blockIdx.x * in_sig_stride)  + (blockIdx.y * in_chan_stride)  + (ref_pol * pol_stride)    + threadIdx.x;
  uint64_t odx = (blockIdx.x * out_sig_stride) + (blockIdx.y * out_chan_stride) + (blockIdx.z * pol_stride) + threadIdx.x;

  __shared__ float normalized_factor;

  for (unsigned iloop=0; iloop<nloops; iloop++)
  {
    // read the reference antenna value
    cuFloatComplex r = in[rdx];

    // read the astronomy antenna value
    cuFloatComplex a = in[idx];

    ///////////////////////////////////
    pa = cuCmulf(a, cuConjf(a));
    pr = cuCmulf(r, cuConjf(r));

    pa = blockReduceSumFC(pa);
    pr = blockReduceSumFC(pr);

    if (threadIdx.x == 0)
    {
      // normalise by the number of values used
      float current_factor = (pa.x / blockDim.x) + (pr.x / blockDim.x);

      if (processed_first_block || iloop > 0)
        normalized_factor = (0.999 * previous_factor) + (0.001 * current_factor);
      else
        normalized_factor = current_factor;
      previous_factor = normalized_factor;
    }

    // ensure normalized factor is shared across the block
    __syncthreads();

    // compute complex conjugate f = [gain * ref]
    cuFloatComplex f = cuCmulf(g, r);

    // subtract from the astronomy signal [af = ast - f]
    cuFloatComplex af = cuCsubf(a, f);

    // compute correlation [corr = af * conj(ref)]
    cuFloatComplex corr = cuCmulf(af, cuConjf(r));

    corr.x /= normalized_factor;
    corr.y /= normalized_factor;

    // sum corr across the block
    corr = blockReduceSumFC(corr);

    // get thread 0 to compute the new gains
    if (threadIdx.x == 0)
    {
      // normalise
      corr.x /= blockDim.x;
      corr.y /= blockDim.x;

      // compute new gain
      const float epsilon = 0.1;
      g.x = (corr.x * epsilon) + g.x;
      g.y = (corr.y * epsilon) + g.y;
    }

    // ensure gains are common across the block
    __syncthreads();

    // now that the gain is updated, for this current block
    f = cuCmulf (g, r);

    // and subtract from the astronomy signal
    af = cuCsubf(a, f);

    // write the output global memory
    out[odx] = af;

    // cleaned power integration
    cleaned_power_sum = cuCmulf(af, cuConjf(af));
    cleaned_power_sum = blockReduceSumFC(cleaned_power_sum);
  
    // increment to the next filter
    idx += blockDim.x;
    rdx += blockDim.x;
    odx += blockDim.x;
  }

  // update the gains
  if (threadIdx.x == 0)
  {
    gains[isigchanpol] = g;
    dirty[isigchanpol] = pa.x / blockDim.x;
    cleaned[isigchanpol] = cleaned_power_sum.x / blockDim.x;
    norms[isigchanpol] = previous_factor;
  }
}


spip::AdaptiveFilterCUDA::AdaptiveFilterCUDA (cudaStream_t _stream, const string& dir) : AdaptiveFilter (dir)
{
  stream = _stream;
  processed_first_block = false;
  gains_file_write = NULL;
  dirty_file_write = NULL;
  cleaned_file_write = NULL;
}

spip::AdaptiveFilterCUDA::~AdaptiveFilterCUDA ()
{
  // ensure the file is closed
  if (gains_file_write)
    gains_file_write->close_file();

  if (dirty_file_write)
    dirty_file_write->close_file();

  if (cleaned_file_write)
    cleaned_file_write->close_file();

  if (gains)
    delete gains;
  gains = NULL;

  if (dirty)
    delete dirty;
  dirty = NULL;

  if (cleaned)
    delete cleaned;
  cleaned = NULL;

  if (norms)
    delete norms;
  norms = NULL;
}

// configure the pipeline prior to runtime
void spip::AdaptiveFilterCUDA::configure (spip::Ordering output_order)
{
  if (!gains)
    gains = new spip::ContainerCUDAFileWrite(output_dir);

  if (!dirty)
    dirty = new spip::ContainerCUDAFileWrite(output_dir);

  if (!cleaned)
    cleaned = new spip::ContainerCUDAFileWrite(output_dir);

  if (!norms)
    norms = new spip::ContainerCUDADevice ();

  spip::AdaptiveFilter::configure (output_order);

  int64_t gains_size = nchan * out_npol * ndim * sizeof(float);
  int64_t dirty_size = nchan * out_npol * sizeof(float);
  int64_t cleaned_size = nchan * out_npol * sizeof(float);

  gains_file_write = dynamic_cast<spip::ContainerCUDAFileWrite *>(gains);
  gains_file_write->set_file_length_bytes (gains_size);
  gains_file_write->process_header ();
  gains_file_write->set_filename_suffix ("gains");

  dirty_file_write = dynamic_cast<spip::ContainerCUDAFileWrite *>(dirty);
  dirty_file_write->set_file_length_bytes (dirty_size);
  dirty_file_write->process_header ();
  dirty_file_write->set_filename_suffix ("dirty");

  cleaned_file_write = dynamic_cast<spip::ContainerCUDAFileWrite *>(cleaned);
  cleaned_file_write->set_file_length_bytes (cleaned_size);
  cleaned_file_write->process_header ();
  cleaned_file_write->set_filename_suffix ("cleaned");
}

// convert to antenna minor order
void spip::AdaptiveFilterCUDA::transform_TSPF()
{
  if (verbose)
    std::cerr << "spip::AdaptiveFilterCUDA::transform_TSPF ()" << endl;
  throw Error (InvalidState, "spip::AdaptiveFilterCUDA::transform_TSPF", "not implemented");
}

void spip::AdaptiveFilterCUDA::transform_SFPT()
{
  if (verbose)
    cerr << "spip::AdaptiveFilterCUDA::transform_SFPT ()" << endl;

  // pointers to the buffers for in, rfi and out
  cuFloatComplex * in = (cuFloatComplex *) input->get_buffer();
  cuFloatComplex * out = (cuFloatComplex *) output->get_buffer();
  cuFloatComplex * gai = (cuFloatComplex *) gains->get_buffer();
  float * dirt = (float *) dirty->get_buffer();
  float * clean = (float *) cleaned->get_buffer();
  float * nor = (float *) norms->get_buffer();

  dim3 blocks (nsignal, nchan, out_npol);
  unsigned nthread = 1024;
  if (nthread > filter_update_time)
    nthread = filter_update_time;

  unsigned nloops = ndat / nthread;
  if (ndat % nthread != 0)
    nloops++;

  if (verbose)
  {
    cerr << "spip::AdaptiveFilterCUDA::transform_SFPT nsignal=" << nsignal 
         << " nchan=" << nchan << " npol=" << npol << " out_npol=" << out_npol << " ref_pol=" << ref_pol << endl;
    cerr << "spip::AdaptiveFilterCUDA::transform_SFPT ndat=" << ndat 
          << " nthread=" << nthread << " nloops=" << nloops << " kernels" << endl;
  }

  uint64_t pol_stride = ndat;
  uint64_t in_chan_stride = npol * pol_stride;
  uint64_t in_sig_stride  = nchan * in_chan_stride;
  uint64_t out_chan_stride = out_npol * pol_stride;
  uint64_t out_sig_stride  = nchan * out_chan_stride;

  if (verbose)
    cerr << "spip::AdaptiveFilterCUDA::transform_SFPT blocks=" << blocks.x 
         << "," << blocks.y << "," << blocks.z << " nthread=" << nthread << endl;

  AdaptiveFilterKernel_SFPT<<<blocks, nthread, 0, stream>>>(in, out, gai, dirt, clean, nor, nloops, processed_first_block, in_sig_stride, in_chan_stride, out_sig_stride, out_chan_stride, pol_stride, ref_pol);

  if (verbose)
    cerr << "spip::AdaptiveFilterCUDA::transform_SFPT kernels complete" << endl;

  processed_first_block = true;
}

// write gains
void spip::AdaptiveFilterCUDA::write_gains ()
{
  uint64_t gains_to_write = (ndat > 0);
  if (verbose)
    cerr << "spip::AdaptiveFilterCUDA::write_gains(" << gains_to_write << ")" << endl;
  gains_file_write->write (gains_to_write);
}

// write dirty
void spip::AdaptiveFilterCUDA::write_dirty ()
{
  uint64_t dirty_to_write = (ndat > 0);
  if (verbose)
    cerr << "spip::AdaptiveFilterCUDA::write_dirty(" << dirty_to_write << ")" << endl;
  dirty_file_write->write (dirty_to_write);
}

// write cleaned
void spip::AdaptiveFilterCUDA::write_cleaned ()
{
  uint64_t cleaned_to_write = (ndat > 0);
  if (verbose)
    cerr << "spip::AdaptiveFilterCUDA::write_cleaned(" << cleaned_to_write << ")" << endl;
  cleaned_file_write->write (cleaned_to_write);
}

