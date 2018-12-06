/***************************************************************************
 *
 *   Copyright (C) 2016 Andrew Jameson
 *   Licensed under the Academic Free License version 2.1
 *
 ***************************************************************************/

#include "spip/CUDAtoCUDATransfer.h"

#include <stdexcept>

using namespace std;

spip::CUDAtoCUDATransfer::CUDAtoCUDATransfer (cudaStream_t _stream) : Transformation<ContainerCUDADevice,ContainerCUDADevice>("CUDAtoCUDATransfer", outofplace)
{
  stream = _stream;
  nblock_out = 1;
  iblock_out = 0;
}

spip::CUDAtoCUDATransfer::~CUDAtoCUDATransfer ()
{
}

void spip::CUDAtoCUDATransfer::set_output_reblock (unsigned factor)
{
  nblock_out = factor;
}

//! intial configuration at the start of the data stream
void spip::CUDAtoCUDATransfer::configure (spip::Ordering output_order)
{
  if (verbose)
    cerr << "spip::CUDAtoCUDATransfer::configure input=" << (void*) input << endl;
  ndat = input->get_ndat ();

  if (verbose)
    cerr << "spip::CUDAtoCUDATransfer::configure: output->clone_header" << endl;
  // copy input header to output
  output->clone_header (input->get_header());

  // output will read the newly cloned header parameters
  output->read_header ();

  // update the output header parameters with the new details
  output->write_header ();
  
  // resize the output container
  prepare_output();

  iblock_out = 0;
}

void spip::CUDAtoCUDATransfer::prepare ()
{
  ndat  = input->get_ndat ();
  if (verbose)
    cerr << "spip::CUDAtoCUDATransfer::prepare ndat=" << ndat << endl;
}

//! simply copy input buffer to output buffer
void spip::CUDAtoCUDATransfer::transformation ()
{
  if (verbose)
    cerr << "spip::CUDAtoCUDATransfer::transformation()" << endl;

  // ensure output is appropriately sized
  prepare_output ();

  uint64_t block_stride = input->get_size();

  void * d_in = (void *) input->get_buffer();
  void * d_out = (void *) (output->get_buffer() + (iblock_out * block_stride));
  size_t nbytes = input->calculate_buffer_size();

  if (verbose)
    cerr << "spip::CUDAtoCUDATransfer::transformation cudaMemcpyAsync (" << (void *) d_out << ", "
         << (void *) d_in << ", " << nbytes << " cudaMemcpyDeviceToDevice, stream)" << endl;

  // perform host to device transfer TODO check for smaller buffers
  cudaError_t err = cudaMemcpyAsync (d_out, d_in, nbytes, cudaMemcpyDeviceToDevice, stream);
  if (err != cudaSuccess)
    throw Error(InvalidState, "spip::CUDAtoCUDATransfer::transformation", cudaGetErrorString (err));

  // increment the output reblocking factor
  iblock_out++;

  // reset once the ouput is full
  if (iblock_out == nblock_out)
  {
    iblock_out = 0;
  }
}

void spip::CUDAtoCUDATransfer::prepare_output ()
{
  output->set_ndat (ndat * nblock_out);
  output->resize();
}
