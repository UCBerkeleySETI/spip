/***************************************************************************
 *
 *   Copyright (C) 2018 Andrew Jameson
 *   Licensed under the Academic Free License version 2.1
 *
 ***************************************************************************/

#include "spip/RINGtoCUDATransfer.h"

#include <stdexcept>

using namespace std;

spip::RINGtoCUDATransfer::RINGtoCUDATransfer (cudaStream_t _stream) : Transformation<ContainerRingRead,ContainerCUDADevice>("RINGtoCUDATransfer", outofplace)
{
  stream = _stream;
  nblock_out = 1;
  iblock_out = 0;
}

spip::RINGtoCUDATransfer::~RINGtoCUDATransfer ()
{
}

void spip::RINGtoCUDATransfer::set_output_reblock (unsigned factor)
{
  nblock_out = factor;
}

//! intial configuration at the start of the data stream
void spip::RINGtoCUDATransfer::configure (spip::Ordering output_order)
{
  ndat  = input->get_ndat ();

  // check location of input ring
  const ContainerRingRead * in = dynamic_cast<const ContainerRingRead*>(input);
  if (!in)
    throw Error(InvalidState, "spip::RINGtoCUDATransfer::configure", "input was not ContainerRingRead");

  // configure the kind of transfer
  if (in->get_db_device() >= 0)
    kind = cudaMemcpyDeviceToDevice;
  else
    kind = cudaMemcpyHostToDevice;

  if (verbose)
    cerr << "spip::RINGtoCUDATransfer::configure: output->clone_header" << endl;
  // copy input header to output
  output->clone_header (input->get_header());

  // output will read the newly cloned header parameters
  output->read_header ();

  // update the output header parameters with the new details
  output->write_header ();
  
  // resize the output container
  prepare_output();

  // 
  iblock_out = 0;
}

void spip::RINGtoCUDATransfer::prepare ()
{
  ndat  = input->get_ndat ();
  if (verbose)
    cerr << "spip::RINGtoCUDATransfer::prepare ndat=" << ndat << endl;
}

//! simply copy input buffer to output buffer
void spip::RINGtoCUDATransfer::transformation ()
{
  if (verbose)
    cerr << "spip::RINGtoCUDATransfer::transformation()" << endl;

  // ensure output is appropriately sized
  prepare_output ();

  uint64_t block_stride = input->get_size();
  if (verbose)
    cerr << "spip::RINGtoCUDATransfer::transformation iblock_out=" << iblock_out << " block_stride=" << block_stride << endl;

  cudaError_t pre_err = cudaGetLastError();
  if (pre_err != cudaSuccess)
    throw Error(InvalidState, "spip::RINGtoCUDATransfer::transformation pre memcpy", cudaGetErrorString (pre_err));

  // handle different input ordering
  if (input->get_order() == spip::Ordering::SFPT)
  {
    float * in_base = (float *) input->get_buffer();
    float * out_base = (float *) output->get_buffer();

    uint64_t out_block_stride = input->get_ndat() * output->get_dat_stride();
    for (unsigned isig=0; isig<input->get_nsignal(); isig++)
    {
      uint64_t in_sig_offset  = isig * input->get_sig_stride();
      uint64_t out_sig_offset = isig * output->get_sig_stride();
      for (unsigned ichan=0; ichan<input->get_nchan(); ichan++)
      {
        uint64_t in_chan_offset  = ichan * input->get_chan_stride();
        uint64_t out_chan_offset = ichan * output->get_chan_stride();
        for (unsigned ipol=0; ipol<input->get_npol(); ipol++)
        {
          uint64_t in_pol_offset  = ipol * input->get_pol_stride();
          uint64_t out_pol_offset = ipol * output->get_pol_stride();

          uint64_t in_offset = in_sig_offset + in_chan_offset + in_pol_offset;
          uint64_t out_offset = out_sig_offset + out_chan_offset + out_pol_offset + (iblock_out * out_block_stride);
   
          void * in = (void *) (in_base + in_offset);
          void * out = (void *) (out_base + out_offset);
          size_t nbytes = out_block_stride * sizeof(float);

          if (verbose)
            cerr << "spip::RINGtoCUDATransfer::transformation cudaMemcpyAsync (" << (void *) out << ", "
                 << (void *) in << ", " << nbytes << ", " << kind << ", stream)" << endl;

          // perform host to device transfer TODO check for smaller buffesr
          cudaError_t err = cudaMemcpyAsync (out, in, nbytes, kind, stream);
          if (err != cudaSuccess)
            throw Error(InvalidState, "spip::RINGtoCUDATransfer::transformation", cudaGetErrorString (err));
        }
      }
    }
  }
  else
  {
    // this works on chars/bytes
    void * in = (void *) input->get_buffer();
    void * out = (void *) (output->get_buffer() + (iblock_out * block_stride));
    size_t nbytes = input->calculate_buffer_size();

    if (verbose)
      cerr << "spip::RINGtoCUDATransfer::transformation cudaMemcpyAsync (" << (void *) out << ", "
           << (void *) in << ", " << nbytes << ", " << kind << ", stream)" << endl;

    // perform host to device transfer TODO check for smaller buffesr
    cudaError_t err = cudaMemcpyAsync (out, in, nbytes, kind, stream);
    if (err != cudaSuccess)
      throw Error(InvalidState, "spip::RINGtoCUDATransfer::transformation", cudaGetErrorString (err));
  }

  // increment the output reblocking factor
  iblock_out++;

  // reset once the ouput is full
  if (iblock_out == nblock_out)
  {
    iblock_out = 0;
  }

  // ensure the copy has ocurred so that the buffer host buffer may be changed
  cudaStreamSynchronize(stream);
}

void spip::RINGtoCUDATransfer::prepare_output ()
{
  output->set_ndat (ndat * nblock_out);
  output->resize();
}
