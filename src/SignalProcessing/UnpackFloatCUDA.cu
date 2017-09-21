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

using namespace std;

//! unpack 8bit integers to floats with ofset and scale
__global__ void unpack_int8(const __restrict__ int32_t* input, float * output, float offset, float scale)
{
  unsigned idx = blockIdx.x * blockDim.x + threadIdx.x;
  unsigned odx = idx * 4;

  // read 4 values at a time
  int32_t packed = input[idx];
  int8_t * packed8 = (int8_t *) &packed;
  
  // TODO check the byte ordering!
  output[odx+0] = (float(packed8[0]) + offset) * scale;
  output[odx+1] = (float(packed8[1]) + offset) * scale;
  output[odx+2] = (float(packed8[2]) + offset) * scale;
  output[odx+3] = (float(packed8[3]) + offset) * scale;
}

//! unpack 16bit integers to floats with ofset and scale
__global__ void unpack_int16(const __restrict__ int32_t* input, float * output, float offset, float scale)
{
  unsigned idx = blockIdx.x * blockDim.x + threadIdx.x;
  unsigned odx = idx * 4;
  
  // read 4 values at a time
  int32_t packed = input[idx]; 
  int16_t * packed16 = (int16_t *) &packed;
  
  // TODO check the byte ordering!
  output[odx+0] = (float(packed16[0]) + offset) * scale;
  output[odx+1] = (float(packed16[1]) + offset) * scale;
}


spip::UnpackFloatCUDA::UnpackFloatCUDA (cudaStream_t _stream)
{
  stream = _stream;
}

spip::UnpackFloatCUDA::~UnpackFloatCUDA ()
{
}

void spip::UnpackFloatCUDA::configure ()
{
  spip::UnpackFloat::configure ();
}

void spip::UnpackFloatCUDA::prepare ()
{
  spip::UnpackFloat::prepare ();
}

void spip::UnpackFloatCUDA::transform ()
{
  if (verbose)
    cerr << "spip::UnpackFloatCUDA::transform" << endl;

  size_t nbytes = input->calculate_buffer_size();

  int nthread = 1024;

  float * out = (float *) output->get_buffer();
  int32_t * in = (int32_t *) input->get_buffer();

  if (nbit == 8)
  {
    int nblock = ndat / (nthread * 4);
    unpack_int8<<<nblock, nthread, 0, stream>>> (in, out, offset, scale);
  }
  else if  (nbit == 16)
  {
    int nblock = ndat / (nthread * 2);
    unpack_int16<<<nblock, nthread, 0, stream>>> (in, out, offset, scale);
  }
  else if (nbit == 32)
  {
    cudaMemcpyAsync ((void *) out, (void *) in, nbytes, cudaMemcpyDeviceToDevice, stream);
  }

}
