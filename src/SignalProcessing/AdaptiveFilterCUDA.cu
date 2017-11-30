/***************************************************************************
 *
 *   Copyright (C) 2016 Andrew Jameson
 *   Licensed under the Academic Free License version 2.1
 *
 ***************************************************************************/

#include "spip/AdaptiveFilterCUDA.h"
#include "spip/Container.h"

#include <iostream>
#include <stdexcept>
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
__global__ void AdaptiveFilterKernel_SFPT (cuFloatComplex * in, cuFloatComplex * ref,
                                           cuFloatComplex * out, cuFloatComplex * gains,
                                           uint64_t nloops, uint64_t sig_stride,
                                           uint64_t chanpol_stride)
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

  //             (isig * sig_stride)       + (ichanpol * chanpol_stride) + sample_idx
  uint64_t idx = (blockIdx.x * sig_stride) + (ichanpol * chanpol_stride) + threadIdx.x;
  //if (threadIdx.x == 0 || threadIdx.x == 1023)
  //  printf ("[%d][%d] ichanpol=%u idx=%lu\n", blockIdx.x, threadIdx.x, ichanpol, idx);

  for (unsigned iloop=0; iloop<nloops; iloop++)
  {
    // read the reference antenna value
    cuFloatComplex r = ref[idx];
    cuFloatComplex a = in[idx];

    // compute complex conjugate f = [gain * ref]
    cuFloatComplex f = cuCmulf(g, r);

    // subtract from the astronomy signal [af = ast - f]
    cuFloatComplex af = cuCsubf(a, f);

    // compute correlation [corr = af * conj(ref)]
    cuFloatComplex corr = cuCmulf(af, cuConjf(r));

    // sum corr across the block
    corr = blockReduceSumFC(corr);

    // get thread 0 to compute the new gains
    if (threadIdx.x == 0)
    {
      // normalise
      corr.x /= blockDim.x;
      corr.y /= blockDim.x;

      // compute new gain
      const float epsilon = 1e-4;
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
    out[idx] = af;
  
    // increment to the next filter
    idx += blockDim.x;
  }

  // update the gains
  if (threadIdx.x == 0)
  {
    gains[ichanpol] = g;
  }
}


spip::AdaptiveFilterCUDA::AdaptiveFilterCUDA ()
{
}

spip::AdaptiveFilterCUDA::~AdaptiveFilterCUDA ()
{
}

void spip::AdaptiveFilterCUDA::set_input_ref (Container * _input_ref)
{
  input_ref = dynamic_cast<spip::ContainerCUDADevice *>(_input_ref);
  if (!input_ref)
    throw Error (InvalidState, "spip::AdaptiveFilterCUDA::set_input_ref",
                 "RFI input was not castable to spip::ContainerCUDADevice *");
}

// configure the pipeline prior to runtime
void spip::AdaptiveFilterCUDA::configure (spip::Ordering output_order)
{
  if (!gains)
    gains = new spip::ContainerCUDADevice ();
  spip::AdaptiveFilter::configure (output_order);
}

//! no special action required
void spip::AdaptiveFilterCUDA::prepare ()
{
  spip::AdaptiveFilter::prepare();
}

// convert to antenna minor order
void spip::AdaptiveFilterCUDA::transform_TSPF()
{
  if (verbose)
    std::cerr << "spip::AdaptiveFilterCUDA::transform_TSPF ()" << endl;
}

void spip::AdaptiveFilterCUDA::transform_SFPT()
{
  if (verbose)
    cerr << "spip::AdaptiveFilterCUDA::transform_SFPT ()" << endl;

  // pointers to the buffers for in, rfi and out
  cuFloatComplex * in = (cuFloatComplex *) input->get_buffer();
  cuFloatComplex * ref = (cuFloatComplex *) input_ref->get_buffer();
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

  uint64_t chanpol_stride = ndat;
  uint64_t sig_stride = nchan * npol * chanpol_stride;

  if (verbose)
    cerr << "spip::AdaptiveFilterCUDA::transform_SFPT chanpol_stride=" << chanpol_stride << " sig_stride=" << sig_stride << endl;

  AdaptiveFilterKernel_SFPT<<<blocks, nthread>>>(in, ref, out, gai, nloops, sig_stride, chanpol_stride);

  if (verbose)
    cerr << "spip::AdaptiveFilterCUDA::transform_SFPT kernels complete" << endl;
}

