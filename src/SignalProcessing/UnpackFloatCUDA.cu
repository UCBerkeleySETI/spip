/***************************************************************************
 *
 *   Copyright (C) 2016 Andrew Jameson
 *   Licensed under the Academic Free License version 2.1
 *
 ***************************************************************************/

#include "spip/UnpackFloatCUDA.h"

#include <iostream>
#include <cstring>
#include <stdexcept>
#include <cmath>
#include <cstdio>

using namespace std;

//! 2 x 16-bit endian swap
__inline__ __device__ int32_t bswap_2_16 (int32_t in) { return  ((in>>8) & 0x00ff00ff ) | ( (in<<8) & 0xff00ff00 ); };

//! 2 x twos complement conversion
__inline__ __device__ int32_t offset_binary_2_16 (int32_t in) { return  in ^ 0x80008000;};

//! unpack 8bit integers to floats with offset and scale
__global__ void unpack_int8(const __restrict__ int32_t* input, float * output, float offset, float scale)
{
  uint64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  uint64_t odx = idx * 4;

  // read 4 values at a time
  int32_t packed = input[idx];
  int8_t * packed8 = (int8_t *) &packed;
  
  // TODO check the byte ordering!
  output[odx+0] = (float(packed8[0]) + offset) * scale;
  output[odx+1] = (float(packed8[1]) + offset) * scale;
  output[odx+2] = (float(packed8[2]) + offset) * scale;
  output[odx+3] = (float(packed8[3]) + offset) * scale;
}

//! unpack 16bit integers to floats with offset and scale [LITTLE ENDIAN, TWOS COMPLEMENT]
__global__ void unpack_int16_littleendian_twoscomplement (const __restrict__ int32_t* input, float * output, float offset, float scale)
{
  uint64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  uint64_t odx = idx * 2;
  
  // read 2 values at a time
  int32_t packed = input[idx]; 
  int16_t * packed16 = (int16_t *) &packed;
  
  // TODO check the byte ordering!
  output[odx+0] = (float(packed16[0]) + offset) * scale;
  output[odx+1] = (float(packed16[1]) + offset) * scale;
}

//! unpack 16bit integers to floats with offset and scale [LITTLE ENDIAN, OFFSET_BINARY]
__global__ void unpack_int16_littleendian_offsetbinary (const __restrict__ int32_t* input, float * output, float offset, float scale)
{
  uint64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  uint64_t odx = idx * 2;

  // read 2 values at a time
  int32_t packed = offset_binary_2_16(input[idx]);
  int16_t * packed16 = (int16_t *) &packed;

  // TODO check the byte ordering!
  output[odx+0] = (float(packed16[0]) + offset) * scale;
  output[odx+1] = (float(packed16[1]) + offset) * scale;
}

//! unpack 16-bit big endian integers to floats with offset and scale
__global__ void unpack_int16_bigendian_twoscomplement(const __restrict__ int32_t* input, float * output, float offset, float scale)
{
  uint64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  uint64_t odx = idx * 2;

  // read 4 values at a time
  int32_t packed = bswap_2_16(input[idx]);
  int16_t * packed16 = (int16_t *) &packed;

  // TODO check the byte ordering!
  output[odx+0] = (float(packed16[0]) + offset) * scale;
  output[odx+1] = (float(packed16[1]) + offset) * scale;
}

//! unpack 16-bit big endian integers to floats with offset and scale
__global__ void unpack_int16_bigendian_offsetbinary (const __restrict__ int32_t* input, float * output, float offset, float scale)
{
  uint64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  uint64_t odx = idx * 2;

  // read 4 values at a time
  int32_t packed = offset_binary_2_16(bswap_2_16(input[idx]));
  int16_t * packed16 = (int16_t *) &packed;

  // TODO check the byte ordering!
  output[odx+0] = (float(packed16[0]) + offset) * scale;
  output[odx+1] = (float(packed16[1]) + offset) * scale;
}


//! 2 x 16-bit endian swap
__inline__ __device__ int32_t bswap_1_32 (int32_t in)
{ 
  return ( ((in>>24) & 0x000000ff) | ((in>>16) & 0x0000ff00) | 
           ((in<<16) & 0x00ff0000) | ((in<<24) & 0xff000000) );
}
 
//! unpack 32-bit big endian integers to floats with offset and scale
__global__ void unpack_int32_bigendian(const __restrict__ int32_t* input, float * output, float offset, float scale)
{
  uint64_t idx = blockIdx.x * blockDim.x + threadIdx.x;

  // read 1 values at a time
  int32_t swapped = bswap_1_32(input[idx]);
  output[idx] = (float(swapped) + offset) * scale;
}



spip::UnpackFloatCUDA::UnpackFloatCUDA (cudaStream_t _stream)
{
  stream = _stream;
}

spip::UnpackFloatCUDA::~UnpackFloatCUDA ()
{
}

void spip::UnpackFloatCUDA::prepare ()
{
  spip::UnpackFloat::prepare ();
}

void spip::UnpackFloatCUDA::transform_SFPT_to_SFPT ()
{
  if (verbose)
    cerr << "spip::UnpackFloatCUDA::transform_SFPT_to_SFPT" << endl;

  size_t nbytes = input->calculate_buffer_size();
  int nthread = 1024;

  float * out = (float *) output->get_buffer();
  int32_t * in = (int32_t *) input->get_buffer();

  if (nbit == 8)
  {
    int nblock = ndim * ndat / (nthread * 4);
    unpack_int8<<<nblock, nthread, 0, stream>>> (in, out, offset, scale);
  }
  else if  (nbit == 16)
  {
    int nblock = ndim * ndat / (nthread * 2);
    if (endianness == spip::Big)
    {
      if (encoding == spip::TwosComplement)
        unpack_int16_bigendian_twoscomplement<<<nblock, nthread, 0, stream>>> (in, out, offset, scale);
      else
        unpack_int16_bigendian_offsetbinary<<<nblock, nthread, 0, stream>>> (in, out, offset, scale);
    }
    else
    {
      if (encoding == spip::TwosComplement)
        unpack_int16_littleendian_twoscomplement<<<nblock, nthread, 0, stream>>> (in, out, offset, scale);
      else
        unpack_int16_littleendian_offsetbinary<<<nblock, nthread, 0, stream>>> (in, out, offset, scale);
    }
  }
  else if (nbit == 32)
  {
    int nblock = ndim * ndat / nthread;
    if (endianness == spip::Big)
      unpack_int32_bigendian<<<nblock, nthread, 0, stream>>> (in, out, offset, scale);
    else
      cudaMemcpyAsync ((void *) out, (void *) in, nbytes, cudaMemcpyDeviceToDevice, stream);
  }

}
