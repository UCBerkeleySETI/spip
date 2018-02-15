/***************************************************************************
 *
 *   Copyright (C) 2017 Andrew Jameson
 *   Licensed under the Academic Free License version 2.1
 *
 ***************************************************************************/

#include "spip/MeerKATPolSubXposeCUDA.h"

#include <cstdio>
#include <cstring>
#include <iostream>
#include <stdexcept>

using namespace std;

spip::MeerKATPolSubXposeCUDA::MeerKATPolSubXposeCUDA (const char * read1_key,
                                                      const char * read2_key,
                                                      const char * write_key,
                                                      int _subband, 
                                                      int _device_id) :
  spip::MeerKATPolSubXpose (read1_key, read2_key, write_key, _subband),
  spip::CudaClient (_device_id)
{
  buffer1 = NULL;
  buffer2 = NULL;
  buffer_size = 0;
}

spip::MeerKATPolSubXposeCUDA::~MeerKATPolSubXposeCUDA ()
{
  cudaError_t error;
  if (buffer1)
    error = cudaFree (buffer1);
  buffer1 = NULL;
  if (buffer2)
    error = cudaFree (buffer2);
  buffer2 = NULL;
}

int64_t spip::MeerKATPolSubXposeCUDA::open ()
{
  return spip::MeerKATPolSubXpose::open();
}
/*
int64_t spip::MeerKATPolSubXposeCUDA::io_block (void * read1_buffer,
                                            void * read2_buffer,
                                            void * write_buffer,
                                            uint64_t write_bytes)
{
  uint64_t nblocks = write_bytes / out_length;
  if (verbose)
    cerr << "spip::MeerKATPolSubXposeCUDA::io_block write_bytes=" << write_bytes 
         << " nblocks=" << nblocks << endl;

  // input buffer pointers
  char * read1 = (char *) read1_buffer;
  char * read2 = (char *) read2_buffer;

  // write buffer pointers
  char * write1 = (char *) write_buffer;
  char * write2 = (char *) write_buffer;

  // increment read pointers by their associated offsets
  read1 += in1_read_offset;
  read2 += in2_read_offset;

  write1 += in1_write_offset;
  write2 += in2_write_offset;

  cudaError_t error;

  // perform the writes from host to device memory
  for (uint64_t iblock=0; iblock<nblocks; iblock++)
  {
    error = cudaMemcpyAsync (write1, read1, in_length, cudaMemcpyHostToDevice, stream);
    if (error != cudaSuccess)
      throw Error (FailedCall, "spip::MeerKATPolSubXposeCUDA::io_block",
                  "cudaMemcpyAsync1(%p, %p, %u): %s", (void *) write1, (void *) read1, in_length,
                   cudaGetErrorString (error));
    error = cudaMemcpyAsync (write2, read2, in_length, cudaMemcpyHostToDevice, stream);
    if (error != cudaSuccess)
      throw Error (FailedCall, "spip::MeerKATPolSubXposeCUDA::io_block",
                  "cudaMemcpyAsync2(%p, %p, %u): %s", (void *) write2, (void *) read2, in_length,
                   cudaGetErrorString (error));

    // increment the input pointers 
    read1 += in_stride;
    read2 += in_stride;

    // increment the output pointers
    write1 += out_stride;
    write2 += out_stride;
  }

  // wait for all copy operations to complete on stream
  cudaStreamSynchronize (stream);

  return int64_t(write_bytes);
}*/

int64_t spip::MeerKATPolSubXposeCUDA::io_block (void * read1_buffer,
                                                void * read2_buffer,
                                                void * write_buffer,
                                                uint64_t write_bytes)
{
  cudaError_t error;

  // adjust the buffer 
  if (buffer_size < write_bytes)
  {
    if (buffer1)
    {
      error = cudaFree (buffer1);
      if (error != cudaSuccess)
        throw Error (FailedCall, "spip::MeerKATPolSubXposeCUDA::io_block",
                   "cudaFree(%p): %s", buffer1, cudaGetErrorString (error));
      buffer1 = NULL;
    }

    if (buffer2)
    {
      error = cudaFree (buffer2);
      if (error != cudaSuccess)
        throw Error (FailedCall, "spip::MeerKATPolSubXposeCUDA::io_block",
                   "cudaFree(%p): %s", buffer2, cudaGetErrorString (error));
      buffer2 = NULL;
    }

    cerr << "spip::MeerKATPolSubXposeCUDA::io_block reallocating buffer1 and buffer1 from "
         << buffer_size << " to " << write_bytes << " bytes" << endl;

    error = cudaMalloc (&buffer1, write_bytes);
    if (error != cudaSuccess)
      throw Error (FailedCall, "spip::MeerKATPolSubXposeCUDA::io_block",
                   "cudaMalloc(%p, %ld): %s", buffer1, write_bytes,
                   cudaGetErrorString (error));

    error = cudaMalloc (&buffer2, write_bytes);
    if (error != cudaSuccess)
      throw Error (FailedCall, "spip::MeerKATPolSubXposeCUDA::io_block",
                   "cudaMalloc(%p, %ld): %s", buffer2, write_bytes,
                   cudaGetErrorString (error));
    buffer_size = write_bytes;
  }

  error = cudaMemcpyAsync (buffer1, read1_buffer, write_bytes, cudaMemcpyHostToDevice, stream);
  if (error != cudaSuccess)
    throw Error (FailedCall, "spip::MeerKATPolSubXposeCUDA::io_block",
                "cudaMemcpyAsync1(%p, %p, %u): %s", buffer1, read1_buffer, write_bytes,
                 cudaGetErrorString (error));

  error = cudaMemcpyAsync (buffer2, read2_buffer, write_bytes, cudaMemcpyHostToDevice, stream);
  if (error != cudaSuccess)
    throw Error (FailedCall, "spip::MeerKATPolSubXposeCUDA::io_block",
                "cudaMemcpyAsync2(%p, %p, %u): %s", buffer2, read2_buffer, write_bytes,
                 cudaGetErrorString (error));

/*
  uint64_t nblocks = write_bytes / out_length;
  if (verbose)
    cerr << "spip::MeerKATPolSubXposeCUDA::io_block write_bytes=" << write_bytes
         << " nblocks=" << nblocks << endl;

  // input buffer pointers
  char * read1 = (char *) buffer1;
  char * read2 = (char *) buffer2;

  // write buffer pointers
  char * write1 = (char *) write_buffer;
  char * write2 = (char *) write_buffer;

  // increment read pointers by their associated offsets
  read1 += in1_read_offset;
  read2 += in2_read_offset;

  write1 += in1_write_offset;
  write2 += in2_write_offset;

  // perform the writes from device to device memory
  for (uint64_t iblock=0; iblock<nblocks; iblock++)
  {
    error = cudaMemcpyAsync (write1, read1, in_length, cudaMemcpyDeviceToDevice, stream);
    if (error != cudaSuccess)
      throw Error (FailedCall, "spip::MeerKATPolSubXposeCUDA::io_block",
                  "cudaMemcpyAsync1(%p, %p, %u): %s", (void *) write1, (void *) read1, in_length,
                   cudaGetErrorString (error));
    error = cudaMemcpyAsync (write2, read2, in_length, cudaMemcpyDeviceToDevice, stream);
    if (error != cudaSuccess)
      throw Error (FailedCall, "spip::MeerKATPolSubXposeCUDA::io_block",
                  "cudaMemcpyAsync2(%p, %p, %u): %s", (void *) write2, (void *) read2, in_length,
                   cudaGetErrorString (error));

    // increment the input pointers 
    read1 += in_stride;
    read2 += in_stride;

    // increment the output pointers
    write1 += out_stride;
    write2 += out_stride;
  }
*/

  // wait for all copy operations to complete on stream
  cudaStreamSynchronize (stream);

  return int64_t(write_bytes);
}


#if 0
  // number of samples per block
  uint64_t nchunks = write_bytes / out_length;
  uint64_t chunk_nsamp = in_length / sizeof(float);

  unsigned nthread = 1024;
  dim3 blocks = (chunk_nsamp/nthreads, nchunks, 1);

  xposeKernel<<<blocks, nthread>>> (buffer1, buffer2, write_buffer, in_stride, out_stride);

}

// SFPT implementation of adaptive filter algorithm
__global__ void xposeKernel (
                              uint64_t chunk_length
                             cuFloatComplex * out, cuFloatComplex * gains,
                             uint64_t nloops, uint64_t sig_stride,
                             uint64_t chanpol_stride)
{
  unsigned iheap = blockIdx.y;
  unsigned isamp  = blockIdx.x * blockDim.x + threadIdx.x;

  uint64_t idx = (iheap * heap_length) + 
  uint64_t odx = 

  // read from buffer1, and buffer2 creating a n
  

}
#endif

int64_t spip::MeerKATPolSubXposeCUDA::close ()
{
  return 0;
}
