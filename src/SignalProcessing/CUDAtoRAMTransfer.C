/***************************************************************************
 *
 *   Copyright (C) 2016 Andrew Jameson
 *   Licensed under the Academic Free License version 2.1
 *
 ***************************************************************************/

#include "spip/CUDAtoRAMTransfer.h"

#include <stdexcept>

using namespace std;

spip::CUDAtoRAMTransfer::CUDAtoRAMTransfer (cudaStream_t _stream) : Transformation<ContainerCUDADevice,Container>("CUDAtoRAMTransfer", outofplace)
{
  stream = _stream;
}

spip::CUDAtoRAMTransfer::~CUDAtoRAMTransfer ()
{
}

//! intial configuration at the start of the data stream
void spip::CUDAtoRAMTransfer::configure (spip::Ordering output_order)
{
  ndat  = input->get_ndat ();

  // copy input header to output
  output->clone_header (input->get_header());

  // output will read the newly cloned header parameters
  output->read_header ();

  // update the output header parameters with the new details
  output->write_header ();
  
  // resize the output container
  prepare_output();
}

void spip::CUDAtoRAMTransfer::prepare ()
{
  ndat  = input->get_ndat ();
  if (verbose)
    cerr << "spip::CUDAtoRAMTransfer::prepare ndat=" << ndat << endl;
}

//! simply copy input buffer to output buffer
void spip::CUDAtoRAMTransfer::transformation ()
{
  if (verbose)
    cerr << "spip::CUDAtoRAMTransfer::transformation()" << endl;

  // ensure output is appropriately sized
  prepare_output ();

  void * host = (void *) input->get_buffer();
  void * device = (void *) output->get_buffer();
  size_t nbytes = input->calculate_buffer_size();

  // perform host to device transfer TODO check for smaller buffesr
  cudaError_t err = cudaMemcpyAsync (device, host, nbytes, cudaMemcpyDeviceToHost, stream);
  if (err != cudaSuccess)
    throw Error(InvalidState, "spip::CUDAtoRAMTransfer::transformation", cudaGetErrorString (err));

  // ensure that the host code can continue from this point
  cudaStreamSynchronize (stream);
}

void spip::CUDAtoRAMTransfer::prepare_output ()
{
  output->set_ndat (ndat);

  output->resize();
}

