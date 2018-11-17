/***************************************************************************
 *
 *   Copyright (C) 2018 Andrew Jameson
 *   Licensed under the Academic Free License version 2.1
 *
 ***************************************************************************/

#include "spip/ReverseFrequencyCUDA.h"

#include <iostream>
#include <cstring>
#include <stdexcept>
#include <cmath>
#include <cstdio>

using namespace std;

spip::ReverseFrequencyCUDA::ReverseFrequencyCUDA (cudaStream_t _stream)
{
  stream = _stream;
}

spip::ReverseFrequencyCUDA::~ReverseFrequencyCUDA ()
{
}

// TSPF ordered kernel, supports in-place operations
__global__ void ReverseFrequency_TSPF_Kernel (float * in, float * out, unsigned nchan,
                                              uint64_t dat_stride, uint64_t sigpol_stride)
{
  const uint64_t channel_base = (blockIdx.z * dat_stride) + (blockIdx.y* sigpol_stride);

  const unsigned ichan_lower = (blockIdx.x * blockDim.x) + threadIdx.x;
  const unsigned ichan_upper = (nchan - ichan_lower) -1;

  if (ichan_lower >= nchan/2)
    return;

  const uint64_t idx_lower = channel_base + ichan_lower;
  const uint64_t idx_upper = channel_base + ichan_upper;

  float in_upper = in[idx_upper];
  out[idx_upper] = in[idx_lower];
  out[idx_lower] = in_upper;
}

// TSPF ordered kernel, supports in-place operations
/*
__global__ void ReverseFrequency_TFPS_Kernel (float * in, float * out, unsigned nchan,
                                              uint64_t dat_stride, uint64_t sigpol_stride)
{
  const uint64_t channel_base = (blockIdx.z * dat_stride) + (blockIdx.y * sigpol_stride);

  const unsigned ichanpolsig_lower = (blockIdx.x * blockDim.x) + threadIdx.x
  const unsigned ichanpolsig_upper = (nchan - ichan_lower) -1;

  if (ichan_lower >= nchan/2)
    return;

  const uint64_t idx_lower = channel_base + ichan_lower;
  const uint64_t idx_upper = channel_base + ichan_upper;

  float in_upper = in[idx_upper];
  out[idx_upper] = in[idx_lower];
  out[idx_lower] = in_upper;
}
*/

void spip::ReverseFrequencyCUDA::transform_TSPF_to_TSPF ()
{
  if (verbose)
    cerr << "spip::ReverseFrequencyCUDA::transform_TSPF_to_TSPF()" << endl;

  if (!reversal)
  {
    transform_copy ();
    return;
  }

  float * in  = (float *) input->get_buffer();
  float * out = (float *) output->get_buffer();

  if (verbose)
  {
    cerr << "spip::ReverseFrequencyCUDA::transform_TSPF_to_TSPF nchan=" << nchan 
         << " nsignal=" << nsignal << " npol=" << npol << " ndat=" << ndat << endl;
  }

  // each thread flips 2 channels
  unsigned nchan_work = nchan / 2;
  unsigned nsigpol = nsignal * npol;
  unsigned nthread = 1024;
  dim3 blocks (nchan_work / nthread, nsigpol, ndat);
  if (nchan_work % nthread != 0)
    blocks.x++;

  uint64_t sigpol_stride = nchan;
  uint64_t dat_stride = sigpol_stride * nsigpol;

  if (verbose) 
  {
    cerr << "spip::ReverseFrequencyCUDA::transform_TSPF_to_TSPF ReverseFrequency_TSPF_FSCR_Kernel" << endl;
    cerr << "blocks=" << blocks.x << "," << blocks.y << "," << blocks.z << " nchan=" << nchan << " nthread=" << nthread << " npol=" << npol << endl;
  }

  ReverseFrequency_TSPF_Kernel<<<blocks, nthread, 0, stream>>> (in, out, nchan, dat_stride, sigpol_stride);
}

void spip::ReverseFrequencyCUDA::transform_TFPS_to_TFPS ()
{
  if (verbose)
    cerr << "spip::ReverseFrequencyCUDA::transform_TFPS_to_TFPS" << endl;

  if (!reversal)
  {
    transform_copy ();
    return;
  }

  float * in  = (float *) input->get_buffer();
  float * out = (float *) output->get_buffer();

  // each thread moves nsigpol samples from 1 channel to another
  //dim3 blocks (nchan_work / nthread, ndat, 1);
  
  //ReverseFrequency_TFPS_Kernel<<<blocks, nthread, 0, stream>>> (in, out, nchan, dat_stride, sigpol_stride);
  throw invalid_argument ("spip::ReverseFrequencyCUDA::transform_TFPS_to_TFPS not implemented (yet)");
}

void spip::ReverseFrequencyCUDA::transform_copy ()
{
  if (verbose)
    cerr << "spip::ReverseFrequencyCUDA::transform_copy" << endl;

  if (this->get_type() == spip::outofplace)
  {
    cerr << "spip::ReverseFrequencyCUDA::transform_copy " << input->get_size() << " bytes" << endl;
    void * in  = (void *) input->get_buffer();
    void * out = (void *) output->get_buffer();

    cudaError_t err = cudaMemcpyAsync (out, in, input->get_size(), cudaMemcpyDeviceToDevice, stream);
    if (err != cudaSuccess)
      throw Error(InvalidState, "spip::ReverseFrequencyCUDA::transform_copy", cudaGetErrorString (err));
  }

  return;
}

