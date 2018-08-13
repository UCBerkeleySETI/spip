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
                                           uint64_t nloops, 
                                           uint64_t in_sig_stride, uint64_t in_chan_stride, 
                                           uint64_t out_sig_stride, uint64_t out_chan_stride,
                                           uint64_t pol_stride, unsigned ref_pol)
{
  // there is one complex gain per block
  __shared__ cuFloatComplex g;

  const unsigned ichanpol = (blockIdx.z * gridDim.y) + blockIdx.y;

  // get the first thread to read the previously computed complex gain
  if (threadIdx.x == 0)
  {
    g = gains[ichanpol];
  }

  // ensure all threads have the complex gain
  __syncthreads();

  // offsets for input, reference and output
  uint64_t idx = (blockIdx.x * in_sig_stride)  + (blockIdx.y * in_chan_stride)  + (blockIdx.z * pol_stride) + threadIdx.x;
  uint64_t rdx = (blockIdx.x * in_sig_stride)  + (blockIdx.y * in_chan_stride)  + (ref_pol * pol_stride)    + threadIdx.x;
  uint64_t odx = (blockIdx.x * out_sig_stride) + (blockIdx.y * out_chan_stride) + (blockIdx.z * pol_stride) + threadIdx.x;

  //if (threadIdx.x == 0 || threadIdx.x == 1023)
  //  printf ("[%d][%d] ichanpol=%u idx=%lu\n", blockIdx.x, threadIdx.x, ichanpol, idx);

  __shared__ float normalized_factor;
  float previos_factor;

  for (unsigned iloop=0; iloop<nloops; iloop++)
  {
    // read the reference antenna value
    cuFloatComplex r = in[rdx];
    // read the astronomy antenna value
    cuFloatComplex a = in[idx];

    ///////////////////////////////////
    cuFloatComplex pa = cuCmulf(a, cuConjf(a));
    cuFloatComplex pr = cuCmulf(r, cuConjf(r));

    pa = blockReduceSumFC(pa);
    pr = blockReduceSumFC(pr);

    if (threadIdx.x == 0)
    {
      // normalise
      float pn = pa.x / blockDim.x + pr.x / blockDim.x;
      //normalized_power.y = pa.y / blockDim.y + pr.y / blockDim.y;

    float current_factor = pn;


    if(iloop == 0)
    {
      normalized_factor = 0.999 * current_factor + 0.001 *current_factor;
    }
    else
    {
      normalized_factor = 0.999 * previos_factor + 0.001 *current_factor;
    }

    previos_factor = current_factor;

    }

    // ensure pn are common across the block
    __syncthreads();

//////////////////////////////////

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
  
    // increment to the next filter
    idx += blockDim.x;
    rdx += blockDim.x;
    odx += blockDim.x;
  }

  // update the gains
  if (threadIdx.x == 0)
  {
    gains[ichanpol] = g;
  }
}


spip::AdaptiveFilterCUDA::AdaptiveFilterCUDA (cudaStream_t _stream, string dir) : AdaptiveFilter (dir)
{
  stream = _stream;
}

spip::AdaptiveFilterCUDA::~AdaptiveFilterCUDA ()
{
}

// configure the pipeline prior to runtime
void spip::AdaptiveFilterCUDA::configure (spip::Ordering output_order)
{
  // TODO implement a ContainerCUDADeviceFileWrite class [Nuer]
  if (!gains)
    gains = new spip::ContainerCUDADevice ();

  spip::AdaptiveFilter::configure (output_order);
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

  dim3 blocks (nsignal, nchan, npol);
  unsigned nthread = 1024;
  if (nthread > filter_update_time)
    nthread = filter_update_time;

  unsigned nloops = ndat / nthread;
  if (ndat % nthread != 0)
    nloops++;

  if (verbose)
  {
    cerr << "spip::AdaptiveFilterCUDA::transform_SFPT nsignal=" << nsignal << " nchan=" << nchan << " npol=" << npol
<< endl;
    cerr << "spip::AdaptiveFilterCUDA::transform_SFPT ndat=" << ndat << " nthread=" << nthread << " nloops=" << nloops << " kernels" << endl;
  }

  uint64_t pol_stride = ndat;
  uint64_t in_chan_stride = npol * pol_stride;
  uint64_t in_sig_stride  = nchan * in_chan_stride;
  uint64_t out_chan_stride = out_npol * pol_stride;
  uint64_t out_sig_stride  = nchan * out_chan_stride;

  AdaptiveFilterKernel_SFPT<<<blocks, nthread, 0, stream>>>(in, out, gai, nloops, in_sig_stride, in_chan_stride, out_sig_stride, out_chan_stride, pol_stride, ref_pol);

  if (verbose)
    cerr << "spip::AdaptiveFilterCUDA::transform_SFPT kernels complete" << endl;
}

void spip::AdaptiveFilterCUDA::write_gains ()
{
  throw Error (InvalidState, "spip::AdaptiveFilterCUDA::write_gains", "not implemented");
}
