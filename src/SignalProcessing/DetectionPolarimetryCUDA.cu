/***************************************************************************
 *
 *   Copyright (C) 2016 Andrew Jameson
 *   Licensed under the Academic Free License version 2.1
 *
 ***************************************************************************/

#include "spip/DetectionPolarimetryCUDA.h"

#include <iostream>
#include <cstring>
#include <stdexcept>
#include <cmath>
#include <cuComplex.h>

using namespace std;

spip::DetectionPolarimetryCUDA::DetectionPolarimetryCUDA (cudaStream_t _stream) : spip::DetectionPolarimetry("DetectionPolarimetryCUDA")
{
  stream = _stream;
}

spip::DetectionPolarimetryCUDA::~DetectionPolarimetryCUDA ()
{
}

inline __device__ float4 cuda_cross_detect (float2 p, float2 q)
{
  float4 c;
  c.x = (p.x * p.x) + (p.y * p.y);
  c.y = (q.x * q.x) + (q.y * q.y);
  c.z = (p.x * q.x) + (p.y * q.y);
  c.w = (p.x * q.y) - (p.y * q.x);
  return c;
}

inline __device__ float4 cuda_stokes_detect (float2 p, float2 q)
{
  const float pp = (p.x * p.x) + (p.y * p.y);
  const float qq = (q.x * q.x) + (q.y * q.y);

  float4 c;
  c.x = pp + qq;
  c.y = pp - qq;
  c.z = 2 * ((p.x * q.x) + (p.y * q.y));
  c.w = 2 * ((p.x * q.y) + (p.y * q.x));

  return c;
}

void spip::DetectionPolarimetryCUDA::transform_SFPT_to_SFPT ()
{
  if (verbose)
    cerr << "spip::DetectionPolarimetryCUDA::transform_SFPT_to_SFPT" << endl;
  throw invalid_argument ("spip::DetectionPolarimetryCUDA::transform_SFPT_to_SFPT not implemented (yet)");
}


__global__ void DetectionPolarimetry_TSPF_Coherence_Kernel (
      const __restrict__ cuFloatComplex * in, float * out,
      uint64_t nchan,
      uint64_t in_sigdat_stride, uint64_t out_sigdat_stride)
{
  // channel processed by this thread
  const uint64_t ichan = (blockIdx.x * blockDim.x) + threadIdx.x;

  if (ichan < nchan)
  {
    //              isigdat      signal/dat offset
    uint64_t idx = (blockIdx.y * in_sigdat_stride)  + ichan;
    uint64_t odx = (blockIdx.y * out_sigdat_stride) + ichan;

    // read pol 0 and pol 1 complex samples
    const cuFloatComplex p = in[idx];
    const cuFloatComplex q = in[idx + nchan];

    float4 c = cuda_cross_detect (p, q);

    out[odx + (0 * nchan)] = c.x;
    out[odx + (1 * nchan)] = c.y;
    out[odx + (2 * nchan)] = c.z;
    out[odx + (3 * nchan)] = c.w;
  }
}

__global__ void DetectionPolarimetry_TSPF_Stokes_Kernel (
      const __restrict__ cuFloatComplex * in, float * out,
      uint64_t nchan,
      uint64_t in_sigdat_stride, uint64_t out_sigdat_stride)
{
  // channel processed by this thread
  const uint64_t ichan = (blockIdx.x * blockDim.x) + threadIdx.x;

  if (ichan < nchan)
  {
    //              isigdat      signal/dat offset
    uint64_t idx = (blockIdx.y * in_sigdat_stride)  + ichan;
    uint64_t odx = (blockIdx.y * out_sigdat_stride) + ichan;

    // read pol 0 and pol 1 complex samples
    const cuFloatComplex p = in[idx];
    const cuFloatComplex q = in[idx + nchan];

    float4 c = cuda_stokes_detect (p, q);

    out[odx + (0 * nchan)] = c.x;
    out[odx + (1 * nchan)] = c.y;
    out[odx + (2 * nchan)] = c.z;
    out[odx + (3 * nchan)] = c.w;
  }
}


void spip::DetectionPolarimetryCUDA::transform_SFPT_to_TSPF ()
{
  throw invalid_argument("spip::DetectionPolarimetryCUDA::transform_SFPT_to_TSP not implemented (yet)");
}


void spip::DetectionPolarimetryCUDA::transform_TSPF_to_TSPF ()
{
  if (verbose)
    cerr << "spip::DetectionPolarimetryCUDA::transform_TSPF_to_TSPF()" << endl;

  cuFloatComplex * in  = (cuFloatComplex *) input->get_buffer();
  float * out = (float *) output->get_buffer();

  unsigned nsigdat = nsignal * ndat;
  unsigned nthread = (nchan < 1024) ? nchan : 1024;
  dim3 blocks (nchan/nthread, nsigdat, 1);
  if (nchan % nthread != 0)
    blocks.x ++;

  uint64_t in_sigdat_stride  = npol * nchan;
  uint64_t out_sigdat_stride = 4 * nchan;
  if (state == spip::Signal::Coherence)
  {
    DetectionPolarimetry_TSPF_Coherence_Kernel<<<blocks, nthread, 0, stream>>>(in, out, nchan, in_sigdat_stride, out_sigdat_stride);
  }
  if (state == spip::Signal::Stokes)
  {
    DetectionPolarimetry_TSPF_Stokes_Kernel<<<blocks, nthread, 0, stream>>>(in, out, nchan, in_sigdat_stride, out_sigdat_stride);

  }

}

void spip::DetectionPolarimetryCUDA::transform_TSPFB_to_TSPFB ()
{
  if (verbose)
    cerr << "spip::DetectionPolarimetryCUDA::transform_TSPFB_to_TSPFB()" << endl;
  throw invalid_argument ("spip::DetectionPolarimetryCUDA::transform_TSPFB_to_TSPFB not implemented (yet)");
}


void spip::DetectionPolarimetryCUDA::transform_TFPS_to_TFPS ()
{
  if (verbose)
    cerr << "spip::DetectionPolarimetryCUDA::transform_TFPS_to_TFPS" << endl;
  throw invalid_argument ("spip::DetectionPolarimetryCUDA::transform_TFPS_to_TFPS not implemented (yet)");
}
