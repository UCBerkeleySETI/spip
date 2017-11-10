/***************************************************************************
 *
 *   Copyright (C) 2016 Andrew Jameson
 *   Licensed under the Academic Free License version 2.1
 *
 ***************************************************************************/

#include "spip/AdaptiveFilterCUDA.h"
#include "spip/Container.h"

#include <iostream>
#include <stdexcept>
#include <cmath>
#include <cuComplex.h>

using namespace std;

// SFPT implementation of adaptive filter algorithm
__global__ void AdaptiveFilterKernel_SFPT (cuFloatComplex * in, cuFloatComplex * ref,
                                           cuFloatComplex * out, cuFloatComplex * gains,
                                           unsigned filter_update_time)
{
  const unsigned idx = blockDim.x * blockIdx.x + threadIdx.x;
  const unsigned igain = blockIdx.z * gridDim.y + blockIdx.y;

  // read the input astronomy and reference antenna samples
  cuFloatComplex r = ref[idx];
  cuFloatComplex g = gains[igain];

  cuFloatComplex f = cuCmulf(g, cuConjf(r));
  cuFloatComplex af = cuCsubf(in[idx], f);

  // write the output global memory
  out[idx] = af;

  cuFloatComplex corr = cuCmulf(af, cuConjf(r));

  // sum corr across the block
  //corr.x = blockReduceSumF(corr.x);
  //corr.y = blockReduceSumF(corr.y);
}


spip::AdaptiveFilterCUDA::AdaptiveFilterCUDA ()
{
}

spip::AdaptiveFilterCUDA::~AdaptiveFilterCUDA ()
{
}

void spip::AdaptiveFilterCUDA::set_input_rfi (Container * _input_rfi)
{
  input_rfi = dynamic_cast<spip::ContainerCUDADevice *>(_input_rfi);
  if (!input_rfi)
    throw Error (InvalidState, "spip::AdaptiveFilterCUDA::set_input_rfi",
                 "RFI input was not castable to spip::ContainerCUDADevice *");
}

// configure the pipeline prior to runtime
void spip::AdaptiveFilterCUDA::configure ()
{
  if (!gains)
    gains = new spip::ContainerCUDADevice ();
  spip::AdaptiveFilter::configure ();
}

//! no special action required
void spip::AdaptiveFilterCUDA::prepare ()
{
  spip::AdaptiveFilter::prepare();
}

// convert to antenna minor order
void spip::AdaptiveFilterCUDA::transform_TSPF()
{
  if (verbose)
    std::cerr << "spip::AdaptiveFilterCUDA::transform_TSPF ()" << endl;
}

void spip::AdaptiveFilterCUDA::transform_SFPT()
{
  if (verbose)
    cerr << "spip::AdaptiveFilterCUDA::transform_SFPT ()" << endl;

  // pointers to the buffers for in, rfi and out
  cuFloatComplex * in = (cuFloatComplex *) input->get_buffer();
  cuFloatComplex * ref = (cuFloatComplex *) input_rfi->get_buffer();
  cuFloatComplex * out = (cuFloatComplex *) output->get_buffer();
  cuFloatComplex * gai = (cuFloatComplex *) gains->get_buffer();

  unsigned nthread = 1024;
  if (nthread > filter_update_time)
    nthread = filter_update_time;
  unsigned nblocks = ndat / nthread;
  if (ndat % nthread != 0)
    nblocks++;

  dim3 blocks (nblocks, nchan, npol);

  // adaptive filter
  AdaptiveFilterKernel_SFPT<<<nthread, blocks>>>(in, ref, out, gai, filter_update_time);
}

