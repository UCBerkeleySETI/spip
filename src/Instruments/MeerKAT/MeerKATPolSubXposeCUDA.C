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
}

spip::MeerKATPolSubXposeCUDA::~MeerKATPolSubXposeCUDA ()
{
}

int64_t spip::MeerKATPolSubXposeCUDA::open ()
{
  return spip::MeerKATPolSubXpose::open();
}

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
}

int64_t spip::MeerKATPolSubXposeCUDA::close ()
{
  return 0;
}
