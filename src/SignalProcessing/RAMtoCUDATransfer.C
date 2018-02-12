/***************************************************************************
 *
 *   Copyright (C) 2016 Andrew Jameson
 *   Licensed under the Academic Free License version 2.1
 *
 ***************************************************************************/

#include "spip/RAMtoCUDATransfer.h"

#include <stdexcept>

using namespace std;

spip::RAMtoCUDATransfer::RAMtoCUDATransfer (cudaStream_t _stream) : Transformation<Container,ContainerCUDADevice>("RAMtoCUDATransfer", outofplace)
{
  stream = _stream;
}

spip::RAMtoCUDATransfer::~RAMtoCUDATransfer ()
{
}

//! intial configuration at the start of the data stream
void spip::RAMtoCUDATransfer::configure (spip::Ordering output_order)
{
  ndat  = input->get_ndat ();

  if (verbose)
    cerr << "spip::RAMtoCUDATransfer::configure: output->clone_header" << endl;
  // copy input header to output
  output->clone_header (input->get_header());

  // output will read the newly cloned header parameters
  output->read_header ();

  // update the output header parameters with the new details
  output->write_header ();
  
  // resize the output container
  prepare_output();
}

void spip::RAMtoCUDATransfer::prepare ()
{
  ndat  = input->get_ndat ();
  if (verbose)
    cerr << "spip::RAMtoCUDATransfer::prepare ndat=" << ndat << endl;
}

//! simply copy input buffer to output buffer
void spip::RAMtoCUDATransfer::transformation ()
{
  if (verbose)
    cerr << "spip::RAMtoCUDATransfer::transformation()" << endl;

  // ensure output is appropriately sized
  prepare_output ();

  void * host = (void *) input->get_buffer();
  void * device = (void *) output->get_buffer();
  size_t nbytes = input->calculate_buffer_size();

  // perform host to device transfer TODO check for smaller buffesr
  cudaError_t err = cudaMemcpyAsync (device, host, nbytes, cudaMemcpyHostToDevice, stream);
  if (err != cudaSuccess)
    throw runtime_error("spip::RAMtoCUDATransfer::transformation cudaMemcpyAsync failed");

  // ensure the copy has ocurred so that the buffer host buffer may be changed
  cudaStreamSynchronize(stream);
}

void spip::RAMtoCUDATransfer::prepare_output ()
{
  output->set_ndat (ndat);
  output->resize();
}

