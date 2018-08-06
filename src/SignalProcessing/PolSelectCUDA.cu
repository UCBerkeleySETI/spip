/***************************************************************************
 *
 *   Copyright (C) 2018 Andrew Jameson
 *   Licensed under the Academic Free License version 2.1
 *
 ***************************************************************************/

#include "spip/PolSelectCUDA.h"

#include <iostream>
#include <stdexcept>
#include <cmath>
#include <cuComplex.h>
#include <cstdio>

using namespace std;

// SFPT implementation of adaptive filter algorithm
__global__ void PolSelectKernel_SFPT (float * in, float* out, 
                                      uint64_t in_sig_stride, uint64_t in_chan_stride, 
                                      uint64_t out_sig_stride, uint64_t out_chan_stride,
                                      uint64_t nval, unsigned out_npol)
{
  const uint64_t ival = (blockIdx.x * blockDim.x) + threadIdx.x;
  const unsigned ichan = blockIdx.y;
  const unsigned isig = blockIdx.z;

  if (ival < nval) 
    return;

  // offsets for input and output
  uint64_t idx = (isig * in_sig_stride) + (ichan * in_chan_stride) + ival;
  uint64_t odx = (isig* out_sig_stride) + (ichan* out_chan_stride) + ival;

  for (unsigned ipol=0; ipol<out_npol; ipol++)
  {
    out[odx] = in[idx];
    idx += nval;
    odx += nval;
  }
}


spip::PolSelectCUDA::PolSelectCUDA (cudaStream_t _stream)
{
  stream = _stream;
}

spip::PolSelectCUDA::~PolSelectCUDA ()
{
}

// convert to antenna minor order
void spip::PolSelectCUDA::transform_TSPF()
{
  if (verbose)
    std::cerr << "spip::PolSelectCUDA::transform_TSPF ()" << endl;
}

void spip::PolSelectCUDA::transform_SFPT()
{
  if (verbose)
    cerr << "spip::PolSelectCUDA::transform_SFPT ()" << endl;

  // pointers to the buffers for in out
  float * in = (float *) input->get_buffer();
  float * out = (float *) output->get_buffer();

  uint64_t pol_stride = ndat * ndim;
  uint64_t in_chan_stride  = npol * pol_stride;
  uint64_t out_chan_stride = out_npol * pol_stride;
  uint64_t in_sig_stride  = nchan * in_chan_stride;
  uint64_t out_sig_stride = nchan * out_chan_stride;

  unsigned nthread = 1024;
  dim3 blocks (pol_stride/nthread, nchan, npol);
  if (pol_stride % nthread != 0)
    blocks.x++;

  PolSelectKernel_SFPT<<<blocks, nthread, 0, stream>>>(in, out, in_sig_stride, out_sig_stride, in_chan_stride, out_chan_stride, pol_stride, out_npol);

  if (verbose)
    cerr << "spip::PolSelectCUDA::transform_SFPT kernels complete" << endl;
}
