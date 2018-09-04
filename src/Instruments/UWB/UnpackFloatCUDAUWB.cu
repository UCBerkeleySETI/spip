/***************************************************************************
 *
 *   Copyright (C) 2018 Andrew Jameson
 *   Licensed under the Academic Free License version 2.1
 *
 ***************************************************************************/

#include "spip/UnpackFloatCUDAUWB.h"

#include <iostream>
#include <cstring>
#include <stdexcept>
#include <cmath>

using namespace std;

spip::UnpackFloatCUDAUWB::UnpackFloatCUDAUWB ()
{
  scale = 1e-4;
}

spip::UnpackFloatCUDAUWB::~UnpackFloatCUDAUWB ()
{
}

void spip::UnpackFloatCUDAUWB::prepare ()
{
  spip::UnpackFloat::prepare ();

  if (endianness != Little)
    throw Error(InvalidState, "spip::UnpackFloatCUDAUWB::prepare", "Expecting Little Endian data");
  if (encoding != OffsetBinary)
    throw Error(InvalidState, "spip::UnpackFloatCUDAUWB::prepare", "Expecting Offset Binary data");
  if (ndim != 2)
    throw Error(InvalidState, "spip::UnpackFloatCUDAUWB::prepare", "Expecting complex sampled input");
  if (nchan != 1)
    throw Error(InvalidState, "spip::UnpackFloatCUDAUWB::prepare", "Expecting 1 input channel"); 
  if (nsignal != 1)
    throw Error(InvalidState, "spip::UnpackFloatCUDAUWB::prepare", "Expecting 1 input signal"); 
}

//! 2 x twos complement conversion
__inline__ __device__ int32_t offset_binary_2_16 (int32_t in) { if (in == 0) return 0; else return  in ^ 0x80008000;};

//! unpack 16bit integers to floats with offset and scale [LITTLE ENDIAN, OFFSET_BINARY]
//! requires 1024 threads, 2048 samples per block, 3 pol and 2 dim
__global__ void unpack_uwb_3pol_2dim_2val (const __restrict__ int32_t* input, float2 * output, float offset, float scale, uint64_t ndat)
{
  // sample number
  uint64_t idx = (blockIdx.x * 6144) + threadIdx.x;
  uint64_t odx = (blockIdx.x * 2048) + threadIdx.x;

  // for handling the the packed data
  int32_t packed;
  int16_t * packed16 = (int16_t *) &packed;
  float2 unpacked;

  // pol 0 sample 0
  packed = offset_binary_2_16(input[idx]);
  unpacked.x = (float(packed16[0]) + offset) * scale;
  unpacked.y = (float(packed16[1]) + offset) * scale;
  output[odx] = unpacked;

  idx += blockDim.x;

  // pol 0 sample 1
  packed = offset_binary_2_16(input[idx]);
  unpacked.x = (float(packed16[0]) + offset) * scale;
  unpacked.y = (float(packed16[1]) + offset) * scale;
  output[odx + blockDim.x] = unpacked;

  idx += blockDim.x;
  odx += ndat;

  // pol 1 sample 0
  packed = offset_binary_2_16(input[idx]);
  unpacked.x = (float(packed16[0]) + offset) * scale;
  unpacked.y = (float(packed16[1]) + offset) * scale;
  output[odx] = unpacked;

  idx += blockDim.x;

  // pol 1 sample 1
  packed = offset_binary_2_16(input[idx]);
  unpacked.x = (float(packed16[0]) + offset) * scale;
  unpacked.y = (float(packed16[1]) + offset) * scale;
  output[odx + blockDim.x] = unpacked;

  idx += blockDim.x;
  odx += ndat;

  // pol 2 sample 0
  packed = offset_binary_2_16(input[idx]);
  unpacked.x = (float(packed16[0]) + offset) * scale;
  unpacked.y = (float(packed16[1]) + offset) * scale;
  output[odx] = unpacked;

  idx += blockDim.x;

  // pol 2 sample 1
  packed = offset_binary_2_16(input[idx]);
  unpacked.x = (float(packed16[0]) + offset) * scale;
  unpacked.y = (float(packed16[1]) + offset) * scale;
  output[odx + blockDim.x] = unpacked;
}


//! unpack 16bit integers to floats with offset and scale [LITTLE ENDIAN, OFFSET_BINARY]
//! requires 1024 threads, 2048 samples per block, 2 pol and 2 dim
__global__ void unpack_uwb_2pol_2dim_2val (const __restrict__ int32_t* input, float2 * output, float offset, float scale, uint64_t ndat)
{
  // sample number
  uint64_t idx = (blockIdx.x * 4096) + threadIdx.x;
  uint64_t odx = (blockIdx.x * 2048) + threadIdx.x;

  // for handling the the packed data
  int32_t packed;
  int16_t * packed16 = (int16_t *) &packed;
  float2 unpacked;

  // pol 0 sample 0
  packed = offset_binary_2_16(input[idx]);
  unpacked.x = (float(packed16[0]) + offset) * scale;
  unpacked.y = (float(packed16[1]) + offset) * scale;
  output[odx] = unpacked;

  idx += blockDim.x;

  // pol 0 sample 1
  packed = offset_binary_2_16(input[idx]);
  unpacked.x = (float(packed16[0]) + offset) * scale;
  unpacked.y = (float(packed16[1]) + offset) * scale;
  output[odx + blockDim.x] = unpacked;

  idx += blockDim.x;
  odx += ndat;

  // pol 1 sample 0
  packed = offset_binary_2_16(input[idx]);
  unpacked.x = (float(packed16[0]) + offset) * scale;
  unpacked.y = (float(packed16[1]) + offset) * scale;
  output[odx] = unpacked;

  idx += blockDim.x;

  // pol 1 sample 1
  packed = offset_binary_2_16(input[idx]);
  unpacked.x = (float(packed16[0]) + offset) * scale;
  unpacked.y = (float(packed16[1]) + offset) * scale;
  output[odx + blockDim.x] = unpacked;
}

__global__ void unpack_uwb_1pol_2dim_2val (const __restrict__ int32_t* input, float2 * output, float offset, float scale, uint64_t ndat)
{
  uint64_t idx = (blockIdx.x * 2048) + threadIdx.x;

  // for handling the the packed data
  int32_t packed;
  int16_t * packed16 = (int16_t *) &packed;
  float2 unpacked;

  // sample 0
  packed = offset_binary_2_16(input[idx]);
  unpacked.x = (float(packed16[0]) + offset) * scale;
  unpacked.y = (float(packed16[1]) + offset) * scale;
  output[idx] = unpacked;

  idx += blockDim.x;

  // sample 1
  packed = offset_binary_2_16(input[idx]);
  unpacked.x = (float(packed16[0]) + offset) * scale;
  unpacked.y = (float(packed16[1]) + offset) * scale;
  output[idx] = unpacked;
}


void spip::UnpackFloatCUDAUWB::transform_custom_to_SFPT ()
{
  if (verbose)
    cerr << "spip::UnpackFloatCUDAUWB::transform_custom_to_SFPT offset=" 
         << offset << " scale=" << scale << endl;

  // in/out
  float2 * out = (float2 *) output->get_buffer();
  int32_t * in = (int32_t *) input->get_buffer();

  size_t nbytes = input->calculate_buffer_size();

  // assumptions are nchan==1, npol==2, ndim==2, nsignal==1
  // each block will process 2048 samples from 1 polarisation
  int nthread = 1024;
  int nsamp_per_block = 2048;
  dim3 blocks (ndat / nsamp_per_block, 1, 1);

  if (npol == 1)
    unpack_uwb_1pol_2dim_2val<<<blocks, nthread, 0, stream>>> (in, out, offset, scale, ndat);
  else if (npol == 2)
    unpack_uwb_2pol_2dim_2val<<<blocks, nthread, 0, stream>>> (in, out, offset, scale, ndat);
  else if (npol == 3)
    unpack_uwb_3pol_2dim_2val<<<blocks, nthread, 0, stream>>> (in, out, offset, scale, ndat);
  else
    throw Error(InvalidState, "spip::UnpackFloatCUDAUWB::transform_custom_to_SFPT", "Expecting 1, 2 or 3 polarisations");
}
