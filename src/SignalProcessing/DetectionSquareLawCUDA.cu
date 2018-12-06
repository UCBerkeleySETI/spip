/***************************************************************************
 *
 *   Copyright (C) 2018 Andrew Jameson
 *   Licensed under the Academic Free License version 2.1
 *
 ***************************************************************************/

#include "spip/DetectionSquareLawCUDA.h"

#include <iostream>
#include <cstring>
#include <stdexcept>
#include <cmath>
#include <cstdio>

#include <cuComplex.h>

using namespace std;

spip::DetectionSquareLawCUDA::DetectionSquareLawCUDA (cudaStream_t _stream) : spip::DetectionSquareLaw("DetectionSquareLawCUDA")
{
  stream = _stream;
}

spip::DetectionSquareLawCUDA::~DetectionSquareLawCUDA ()
{
}

__global__ void DetectionSquareLaw_SFPT_PPQQ_Kernel (
      const __restrict__ cuFloatComplex * in, float * out,
      uint64_t ndat, uint64_t sig_stride, uint64_t chan_stride, uint64_t pol_stride)
{
  // sample offset
  const uint64_t idat = (blockIdx.x * blockDim.x) + threadIdx.x;

  if (idat >= ndat)
    return;

  //              signal offset               channel offset
  uint64_t idx = (blockIdx.y * sig_stride) + (blockIdx.z * chan_stride) + idat;

  // polarisation 0 first
  const cuFloatComplex p = in[idx];
  out[idx] = (p.x * p.x) + (p.y * p.y);

  // shift to second polarisation
  idx += pol_stride;

  const cuFloatComplex q = in[idx];
  out[idx] = (q.x * q.x) + (q.y * q.y);
}

__global__ void DetectionSquareLaw_SFPT_Intensity_Kernel (
      const __restrict__ cuFloatComplex * in, float * out,
      uint64_t ndat, uint64_t npol,
      uint64_t sig_stride, uint64_t chan_stride, uint64_t pol_stride)
{
  // sample offset
  const uint64_t idat = (blockIdx.x * blockDim.x) + threadIdx.x;
  
  if (idat >= ndat)
    return;
  
  uint64_t out_block_offset = (blockIdx.y * sig_stride) + (blockIdx.z * chan_stride);

  // compute the output offset
  const uint64_t odx = out_block_offset + idat;

  // the input block offset includes an npol multiplier
  uint64_t idx = (npol * odx) + idat;
  float sum = 0.0f;
 
  for (unsigned ipol=0; ipol<npol; ipol++)
  {
    const cuFloatComplex p = in[idx];
    sum += (p.x * p.x) + (p.y * p.y);
    idx += ndat;
  }
  out[idx] = sum;
}

void spip::DetectionSquareLawCUDA::transform_SFPT_to_SFPT ()
{
  if (verbose)
    cerr << "spip::DetectionSquareLawCUDA::transform_SFPT_to_SFPT" << endl;

  cuFloatComplex * in = (cuFloatComplex *) input->get_buffer();
  float * out = (float *) output->get_buffer();

  if (state == spip::Signal::PPQQ)
  {
    unsigned nthread = 1024;
    dim3 blocks (ndat/nthread, nsignal, nchan);
    if (ndat % nthread != 0)
      blocks.x ++;

    uint64_t pol_stride = ndat;
    uint64_t chan_stride = npol * ndat;
    uint64_t sig_stride = nsignal * chan_stride;

    // npol == 2 by definition
    DetectionSquareLaw_SFPT_PPQQ_Kernel<<<blocks, nthread, 0, stream>>>(in, out, ndat, sig_stride, chan_stride, pol_stride);
  }

  if (state == spip::Signal::Intensity)
  {
    unsigned nthread = 1024;
    dim3 blocks (ndat/nthread, nsignal, nchan);
    if (ndat % nthread != 0)
      blocks.x ++;

    uint64_t pol_stride = ndat;
    uint64_t chan_stride = pol_stride;
    uint64_t sig_stride = nsignal * chan_stride;

    DetectionSquareLaw_SFPT_Intensity_Kernel<<<blocks, nthread, 0, stream>>>(in, out, ndat, npol, sig_stride, chan_stride, pol_stride);
  }
}

__global__ void DetectionSquareLaw_SFPT_to_TSPF_PPQQ_Kernel (
      const __restrict__ cuFloatComplex * in, float * out, uint64_t ndat, 
      uint64_t in_sig_stride, uint64_t in_chan_stride, 
      uint64_t out_dat_stride, uint64_t out_sig_stride)
{
  // sample offset
  const uint64_t idat = (blockIdx.x * blockDim.x) + threadIdx.x;

  if (idat >= ndat)
    return;

  //              signal offset                  channel offset
  uint64_t idx = (blockIdx.y * in_sig_stride) + (blockIdx.z * in_chan_stride) + idat;

  //              dat offset                signal offset                  channel offset 
  uint64_t odx = (idat * out_dat_stride) + (blockIdx.y * out_sig_stride) + blockIdx.z;

  // polarisation 0 first
  const cuFloatComplex p = in[idx];
  out[odx] = (p.x * p.x) + (p.y * p.y);

  // shift to second polarisation
  idx += ndat;      // in_pol_stride
  odx += gridDim.z; // nchan

  const cuFloatComplex q = in[idx];
  out[odx] = (q.x * q.x) + (q.y * q.y);
}

__global__ void DetectionSquareLaw_SFPT_to_TSPF_Intensity_Kernel (
      const __restrict__ cuFloatComplex * in, float * out,
      uint64_t ndat, uint64_t npol,
      uint64_t in_sig_stride, uint64_t in_chan_stride,
      uint64_t out_dat_stride, uint64_t out_sig_stride)
{ 
  // sample offset
  const uint64_t idat = (blockIdx.x * blockDim.x) + threadIdx.x;
  
  if (idat >= ndat)
    return;
  
  //              signal offset                  channel offset
  uint64_t idx = (blockIdx.y * in_sig_stride) + (blockIdx.z * in_chan_stride) + idat;

  //              dat offset                signal offset                  channel offset 
  uint64_t odx = (idat * out_dat_stride) + (blockIdx.y * out_sig_stride) + blockIdx.z;

  float sum = 0.0f;
  for (unsigned ipol=0; ipol<npol; ipol++)
  {
    const cuFloatComplex p = in[idx];
    sum += (p.x * p.x) + (p.y * p.y);
    idx += ndat;
  }

  out[odx] = sum;  
}

void spip::DetectionSquareLawCUDA::transform_SFPT_to_TSPF ()
{
  if (verbose)
    cerr << "spip::DetectionSquareLawCUDA::transform_SFPT_to_TSPF" << endl;

  cuFloatComplex * in = (cuFloatComplex *) input->get_buffer();
  float * out = (float *) output->get_buffer();

  uint64_t pol_stride = ndat;
  uint64_t in_chan_stride = npol * pol_stride;
  uint64_t in_sig_stride = nsignal * in_chan_stride;

  unsigned nthread = 1024;
  dim3 blocks (ndat/nthread, nsignal, nchan);
  if (ndat % nthread != 0)
    blocks.x ++;
    
  if (state == spip::Signal::PPQQ)
  { 
    uint64_t out_sig_stride = nchan * npol;
    uint64_t out_dat_stride = nsignal * out_sig_stride;
    if (verbose)
    {
      cerr << "spip::DetectionSquareLawCUDA::transform_SFPT_to_TSPF blocks=(" << blocks.x << "," << blocks.y << "," << blocks.z << ")" << endl;
    }
  
    // npol == 2 by definition
    DetectionSquareLaw_SFPT_to_TSPF_PPQQ_Kernel<<<blocks, nthread, 0, stream>>>(in, out, ndat, in_sig_stride, in_chan_stride, out_dat_stride, out_sig_stride);
  }
    
  if (state == spip::Signal::Intensity) 
  {
    uint64_t out_sig_stride = nchan;
    uint64_t out_dat_stride = nsignal * out_sig_stride;

    DetectionSquareLaw_SFPT_to_TSPF_Intensity_Kernel<<<blocks, nthread, 0, stream>>>(in, out, ndat, npol, in_sig_stride, in_chan_stride, out_dat_stride, out_sig_stride);

  }
}


__global__ void DetectionSquareLaw_TSPF_PPQQ_Kernel (
      const __restrict__ cuFloatComplex * in, float * out,
      uint64_t nchan, 
      uint64_t dat_stride, uint64_t sig_stride, uint64_t pol_stride)
{
  // channel processed by this thread
  const uint64_t ichan = (blockIdx.x * blockDim.x) + threadIdx.x;

  if (ichan < nchan)
  {
    const unsigned isig = blockIdx.y;
    const unsigned idat = blockIdx.z;

    //              signal offset         dat offset
    uint64_t idx = (isig * sig_stride) + (idat * dat_stride) + ichan;

    // polarisation 0 first
    const cuFloatComplex p = in[idx];
    out[idx] = (p.x * p.x) + (p.y * p.y);

    // shift to second polarisation
    idx += pol_stride;

    const cuFloatComplex q = in[idx];
    out[idx] = (q.x * q.x) + (q.y * q.y);
  }
}

// each thread in a block reads a different channel
__global__ void DetectionSquareLaw_TSPF_Intensity_Kernel (
      const __restrict__ cuFloatComplex * in, float * out,
      uint64_t nchan, unsigned npol,
      uint64_t in_dat_stride, uint64_t out_dat_stride,
      uint64_t in_sig_stride, uint64_t out_sig_stride, 
      uint64_t pol_stride)
{
  // channel processed by this thread
  const uint64_t ichan = (blockIdx.x * blockDim.x) + threadIdx.x;

  if (ichan < nchan)
  {
    const unsigned isig = blockIdx.y;
    const unsigned idat = blockIdx.z;

    uint64_t idx       = (idat * in_dat_stride)  + (isig * in_sig_stride)  + ichan;
    const uint64_t odx = (idat * out_dat_stride) + (isig * out_sig_stride) + ichan;

    float sum = 0.0f;
    for (unsigned ipol=0; ipol<npol; ipol++)
    {
      const cuFloatComplex p = in[idx];
      sum += (p.x * p.x) + (p.y * p.y);
      idx += pol_stride;
    }
    out[odx] = sum;
    //out[odx] = float(ichan);
  }
}

__global__ void DetectionSquareLaw_TSPFB_PPQQ_Kernel (
      const __restrict__ cuFloatComplex * in, float * out,
      uint64_t nchan,
      uint64_t dat_stride, uint64_t sig_stride, uint64_t pol_stride)
{
  // channel processed by this thread
  const uint64_t ichan = (blockIdx.x * blockDim.x) + threadIdx.x;

  if (ichan < nchan)
  {
    const unsigned isig = blockIdx.y;
    const unsigned idat = blockIdx.z;

    //              signal offset         dat offset
    uint64_t idx = (isig * sig_stride) + (idat * dat_stride) + ichan;

    // polarisation 0 first
    const cuFloatComplex p = in[idx];
    out[idx] = (p.x * p.x) + (p.y * p.y);

    // shift to second polarisation
    idx += pol_stride;

    const cuFloatComplex q = in[idx];
    out[idx] = (q.x * q.x) + (q.y * q.y);
  }
}

// each thread in a block reads a different channel
__global__ void DetectionSquareLaw_TSPFB_Intensity_Kernel (
      const __restrict__ cuFloatComplex * in, float * out,
      uint64_t nchan, unsigned npol,
      uint64_t in_dat_stride, uint64_t out_dat_stride,
      uint64_t in_sig_stride, uint64_t out_sig_stride,
      uint64_t pol_stride)
{
  // channel processed by this thread
  const uint64_t ichan = (blockIdx.x * blockDim.x) + threadIdx.x;

  if (ichan < nchan)
  {
    const unsigned isig = blockIdx.y;
    const unsigned idat = blockIdx.z;

    uint64_t idx       = (idat * in_dat_stride)  + (isig * in_sig_stride)  + ichan;
    const uint64_t odx = (idat * out_dat_stride) + (isig * out_sig_stride) + ichan;

    float sum = 0.0f;
    for (unsigned ipol=0; ipol<npol; ipol++)
    {
      const cuFloatComplex p = in[idx];
      sum += (p.x * p.x) + (p.y * p.y);
      idx += pol_stride;
    }
    out[odx] = sum;
  }
}

void spip::DetectionSquareLawCUDA::transform_TSPF_to_TSPF ()
{
  if (verbose)
    cerr << "spip::DetectionSquareLawCUDA::transform_TSPF_to_TSPF()" << endl;

  cuFloatComplex * in  = (cuFloatComplex *) input->get_buffer();
  float * out = (float *) output->get_buffer();

  unsigned nthread = (nchan < 1024) ? nchan : 1024;
  dim3 blocks (nchan/nthread, nsignal, ndat);
  if (nchan % nthread != 0)
    blocks.x ++;

  const uint64_t pol_stride = nchan;

  if (state == spip::Signal::PPQQ)
  {
    uint64_t sig_stride = npol * pol_stride;
    uint64_t dat_stride = nsignal * sig_stride;

    if (verbose)
    {
      cerr << "spip::DetectionSquareLawCUDA::transform_TSPF_to_TSPF DetectionSquareLaw_TSPF_PPQQ_Kernel" << endl;
      cerr << "spip::DetectionSquareLawCUDA::transform_TSPF_to_TSPF pol_stride=" << pol_stride << " sig_stride=" << sig_stride << " dat_stride=" << dat_stride << endl;
    }
    DetectionSquareLaw_TSPF_PPQQ_Kernel<<<blocks, nthread, 0, stream>>>(in, out, nchan, dat_stride, sig_stride, pol_stride);
  }

  if (state == spip::Signal::Intensity)
  {
    uint64_t in_sig_stride = npol * pol_stride;
    uint64_t out_sig_stride = pol_stride;
    uint64_t in_dat_stride = nsignal * in_sig_stride;
    uint64_t out_dat_stride = nsignal * out_sig_stride;

    if (verbose)
    {
      cerr << "spip::DetectionSquareLawCUDA::transform_TSPF_to_TSPF DetectionSquareLaw_TSPF_Intensity_Kernel()" << endl;
      cerr << "spip::DetectionSquareLawCUDA::transform_TSPF_to_TSPF pol_stride=" << pol_stride << endl;
      cerr << "spip::DetectionSquareLawCUDA::transform_TSPF_to_TSPF in_sig_stride=" << in_sig_stride << " out_sig_stride=" << out_sig_stride << endl;
      cerr << "spip::DetectionSquareLawCUDA::transform_TSPF_to_TSPF in_dat_stride=" << in_dat_stride << " out_dat_stride=" << out_dat_stride << endl;
      cerr << "spip::DetectionSquareLawCUDA::transform_TSPF_to_TSPF blocks=" << blocks.x << "," << blocks.y << "," << blocks.z << " nchan=" << nchan << " nthread=" << nthread << " npol=" << npol << endl;
    }
    
    DetectionSquareLaw_TSPF_Intensity_Kernel<<<blocks, nthread, 0, stream>>>(in, out, nchan, npol, in_dat_stride, out_dat_stride,
                                                                             in_sig_stride, out_sig_stride, pol_stride);
  }
}

void spip::DetectionSquareLawCUDA::transform_TSPFB_to_TSPFB ()
{
  if (verbose)
    cerr << "spip::DetectionSquareLawCUDA::transform_TSPFB_to_TSPFB()" << endl;

  cuFloatComplex * in  = (cuFloatComplex *) input->get_buffer();
  float * out = (float *) output->get_buffer();

  unsigned nthread = (nchan < 1024) ? nchan : 1024;
  dim3 blocks (nchan/nthread, nsignal, ndat);
  if (nchan % nthread != 0)
    blocks.x ++;

  const uint64_t pol_stride = nchan;

  if (state == spip::Signal::PPQQ)
  {
    uint64_t sig_stride = npol * pol_stride;
    uint64_t dat_stride = nsignal * sig_stride;

    cerr << "spip::DetectionSquareLawCUDA::transform_TSPFB_to_TSPFB DetectionSquareLaw_TSPFB_PPQQ_Kernel()" << endl;
    DetectionSquareLaw_TSPFB_PPQQ_Kernel<<<blocks, nthread, 0, stream>>>(in, out, nchan, dat_stride, sig_stride, pol_stride);
  }

  if (state == spip::Signal::Intensity)
  {
    uint64_t in_sig_stride = npol * pol_stride;
    uint64_t out_sig_stride = pol_stride;
    uint64_t in_dat_stride = nsignal * in_sig_stride;
    uint64_t out_dat_stride = nsignal * out_sig_stride;

    if (verbose)
    {
      cerr << "spip::DetectionSquareLawCUDA::transform_TSPFB_to_TSPFB DetectionSquareLaw_TSPFB_Intensity_Kernel()" << endl;
      cerr << "spip::DetectionSquareLawCUDA::transform_TSPFB_to_TSPFB pol_stride=" << pol_stride << endl;
      cerr << "spip::DetectionSquareLawCUDA::transform_TSPFB_to_TSPFB in_sig_stride=" << in_sig_stride << " out_sig_stride=" << out_sig_stride << endl;
      cerr << "spip::DetectionSquareLawCUDA::transform_TSPFB_to_TSPFB in_dat_stride=" << in_dat_stride << " out_dat_stride=" << out_dat_stride << endl;
      cerr << "spip::DetectionSquareLawCUDA::transform_TSPFB_to_TSPFB blocks=" << blocks.x << "," << blocks.y << "," << blocks.z << " nchan=" << nchan << " nthread=" << nthread << " npol=" << npol << endl;
    }

    DetectionSquareLaw_TSPFB_Intensity_Kernel<<<blocks, nthread, 0, stream>>>(in, out, nchan, npol, in_dat_stride, out_dat_stride,
                                                                             in_sig_stride, out_sig_stride, pol_stride);
  }
}

void spip::DetectionSquareLawCUDA::transform_TFPS_to_TFPS ()
{
  if (verbose)
    cerr << "spip::DetectionSquareLawCUDA::transform_TFPS_to_TFPS" << endl;
  throw invalid_argument ("spip::DetectionSquareLawCUDA::transform_TFPS_to_TFPS not implemented (yet)");
}
