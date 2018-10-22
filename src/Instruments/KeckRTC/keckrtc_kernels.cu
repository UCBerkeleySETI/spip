/***************************************************************************
 *
 *   Copyright (C) 2018 Andrew Jameson
 *   Licensed under the Academic Free License version 2.1
 *
 ***************************************************************************/

#include "spip/keckrtc_kernels.h"


__global__ void keckrtc_dummy_kernel (int8_t * data, size_t nbytes)
{
  int idx = blockIdx.x * blockDim.x + threadIdx.x;

  if (idx >= nbytes)
    return;

  data[idx] = 2 * data[idx];
}

void keckrtc_dummy (void * dev_buf, size_t bufsz, cudaStream_t stream)
{
  int8_t * data = (int8_t *) dev_buf;

  int nthread = 1024;
  int nblocks = bufsz / nthread;
  if (bufsz % nthread)
    nblocks++;

  keckrtc_dummy_kernel<<<nblocks, nthread, 0, stream>>> (data, bufsz);
}

