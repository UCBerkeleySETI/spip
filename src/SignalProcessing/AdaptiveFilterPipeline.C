/***************************************************************************
 *
 *   Copyright (C) 2016 Andrew Jameson
 *   Licensed under the Academic Free License version 2.1
 *
 ***************************************************************************/

#include "spip/AdaptiveFilterPipeline.h"

#include <signal.h>
#include <unistd.h>
#include <cstring>
#include <iostream>
#include <stdexcept>
#include <new>

//#define _DEBUG

using namespace std;

spip::AdaptiveFilterPipeline::AdaptiveFilterPipeline (const char * in_key_string, const char * ref_key_string, const char * out_key_string)
{
  nfft = -1;
  device = -1;

  in_db  = new DataBlockRead (in_key_string);
  ref_db  = new DataBlockRead (ref_key_string);
  out_db = new DataBlockWrite (out_key_string);

  in_db->connect();
  in_db->lock();

  ref_db->connect();
  ref_db->lock();

  out_db->connect();
  out_db->lock();

  verbose = false;
}

spip::AdaptiveFilterPipeline::~AdaptiveFilterPipeline()
{
  in_db->unlock();
  in_db->disconnect();
  delete in_db;

  ref_db->unlock();
  ref_db->disconnect();
  delete ref_db;

  out_db->unlock();
  out_db->disconnect();
  delete out_db;
}

void spip::AdaptiveFilterPipeline::set_channelisation (int freq_res)
{
  if (verbose)
    cerr << "spip::AdaptiveFilterPipeline::set_channelisation freq_res=" << freq_res  << endl;
  nfft = freq_res;
}

//! build the pipeline containers and transforms
void spip::AdaptiveFilterPipeline::configure (spip::UnpackFloat * unpacker, spip::UnpackFloat * unpacker_ref)
{
  if (verbose)
    cerr << "spip::AdaptiveFilterPipeline::configure ()" << endl;
#ifdef HAVE_CUDA
  if (device >= 0)
    return configure_cuda(unpacker, unpacker_ref);
#endif
  
  if (verbose)
    cerr << "spip::AdaptiveFilterPipeline::configure creating input" << endl;
  // input containers, reads header 
  input = new spip::ContainerRingRead (in_db);
  input_ref = new spip::ContainerRingRead (ref_db);

  if (verbose)
    cerr << "spip::AdaptiveFilterPipeline::configure unpacked container" << endl;
  // unpacked containers
  unpacked = new spip::ContainerRAM ();
  unpacked_ref = new spip::ContainerRAM ();
  
  if (verbose)
    cerr << "spip::AdaptiveFilterPipeline::configure allocating UnpackFloat" << endl;
  // unpack to float operation
  unpack_float = unpacker;
  unpack_float->set_input (input);
  unpack_float->set_output (unpacked);
  unpack_float->set_verbose (verbose);

  unpack_float_ref = unpacker_ref;
  unpack_float_ref->set_input (input_ref);
  unpack_float_ref->set_output (unpacked_ref);
  unpack_float_ref->set_verbose (verbose);

  // fine channels
  if (verbose)
    cerr << "spip::AdaptiveFilterPipeline::configure allocating channelised container" << endl;
  channelised = new spip::ContainerRAM ();
  channelised_ref = new spip::ContainerRAM ();
  
  if (verbose)
    cerr << "spip::AdaptiveFilterPipeline::configure allocating Forward FFT" << endl;
  // forward FFT operation
  fwd_fft = new spip::ForwardFFTFFTW();
  fwd_fft->set_input (unpacked);
  fwd_fft->set_output (channelised);
  fwd_fft->set_nfft (nfft);
  fwd_fft->set_verbose (verbose);

  fwd_fft_ref = new spip::ForwardFFTFFTW();
  fwd_fft_ref->set_input (unpacked_ref);
  fwd_fft_ref->set_output (channelised_ref);
  fwd_fft_ref->set_nfft (nfft);
  fwd_fft_ref->set_verbose (verbose);

  // cleaned data
  cleaned = new spip::ContainerRAM ();

  // RFI Filtering operation
  filter = new spip::AdaptiveFilterRAM();
  filter->set_input (channelised);
  filter->set_input_ref (channelised_ref);
  filter->set_output (cleaned);
  filter->set_verbose (verbose);

  if (verbose)
    cerr << "spip::AdaptiveFilterPipeline::configure allocating output Ring Buffer" << endl;
  // coarse output channels
  output = new spip::ContainerRingWrite (out_db);
  
  if (verbose)
    cerr << "spip::AdaptiveFilterPipeline::configure allocating Backward FFT" << endl;
  // backward fft operation
  bwd_fft = new spip::BackwardFFTFFTW();
  bwd_fft->set_input (cleaned);
  bwd_fft->set_output (output);
  bwd_fft->set_nfft (nfft);
  bwd_fft->set_verbose (verbose);
}

#ifdef HAVE_CUDA

void spip::AdaptiveFilterPipeline::set_device (int _device)
{
  int device_count = 0;
  cudaError_t err = cudaGetDeviceCount(&device_count);
  if (err != cudaSuccess)
    throw runtime_error ("spip::AdaptiveFilterPipeline::set_device cudaGetDeviceCount failed");

  device = _device;
  if (device >= device_count)
    throw runtime_error ("Specified CUDA device not available");

  // TODO error checking
  err = cudaSetDevice (device);
  if (err != cudaSuccess)
    throw runtime_error ("spip::AdaptiveFilterPipeline::set_device cudaSetDevice failed");

  err = cudaStreamCreate(&stream);
   if (err != cudaSuccess)
    throw runtime_error ("spip::AdaptiveFilterPipeline::set_device cudaStreamCreate failed");
}
  
//! build the pipeline containers and transforms
void spip::AdaptiveFilterPipeline::configure_cuda (spip::UnpackFloat * unpacker, spip::UnpackFloat * unpacker_ref)
{
  if (verbose)
    cerr << "spip::AdaptiveFilterPipeline::configure_cuda creating input" << endl;
  // input container, reads header 
  input = new spip::ContainerRingRead (in_db);
  input->register_buffers();

  input_ref = new spip::ContainerRingRead (ref_db);
  input_ref->register_buffers();

  // transfer host to device
  d_input = new spip::ContainerCUDADevice ();
  d_input_ref = new spip::ContainerCUDADevice ();

  if (verbose)
    cerr << "spip::AdaptiveFilterPipeline::configure_cuda allocating RAM to CUDA Transfer" << endl;
  ram_to_cuda = new spip::RAMtoCUDATransfer (stream);
  ram_to_cuda->set_input (input);
  ram_to_cuda->set_output (d_input); 
  ram_to_cuda->set_verbose (verbose);

  ram_to_cuda_ref = new spip::RAMtoCUDATransfer (stream);
  ram_to_cuda_ref->set_input (input_ref);
  ram_to_cuda_ref->set_output (d_input_ref);
  ram_to_cuda_ref->set_verbose (verbose);

  if (verbose)
    cerr << "spip::AdaptiveFilterPipeline::configure_cuda unpacked container" << endl;
  // unpacked container
  unpacked = new spip::ContainerCUDADevice ();
  unpacked_ref = new spip::ContainerCUDADevice ();
  if (verbose)
    cerr << "spip::AdaptiveFilterPipeline::configure_cuda allocating UnpackFloat" << endl;

  // unpack to float operation
  unpack_float = unpacker;
  unpack_float->set_input (d_input);
  unpack_float->set_output (unpacked);
  unpack_float->set_verbose (verbose);
  UnpackFloatCUDA * tmp = dynamic_cast<UnpackFloatCUDA *>(unpacker);
  if (tmp)
    tmp->set_stream (stream);
  else
    throw Error (InvalidState, "spip::AdaptiveFilterPipeline::configure_cuda", "unpacker must be a UnpackFloatCUDA");

  unpack_float_ref = unpacker_ref;
  unpack_float_ref->set_input (d_input_ref);
  unpack_float_ref->set_output (unpacked_ref);
  unpack_float_ref->set_verbose (verbose);
  tmp = dynamic_cast<UnpackFloatCUDA *>(unpacker_ref);
  if (tmp)
    tmp->set_stream (stream);
  else
    throw Error (InvalidState, "spip::AdaptiveFilterPipeline::configure_cuda", "unpacker must be a UnpackFloatCUDA");

  // fine channels
  if (verbose)
    cerr << "spip::AdaptiveFilterPipeline::configure_cuda allocating channelised container" << endl;
  channelised = new spip::ContainerCUDADevice ();
  channelised_ref = new spip::ContainerCUDADevice ();

  if (verbose)
    cerr << "spip::AdaptiveFilterPipeline::configure_cuda allocating Forward FFT" << endl;
  // forward FFT operation
  fwd_fft = new spip::ForwardFFTCUDA(stream);
  fwd_fft->set_input (unpacked);
  fwd_fft->set_output (channelised);
  fwd_fft->set_nfft (nfft);
  fwd_fft->set_verbose (verbose);

  fwd_fft_ref = new spip::ForwardFFTCUDA(stream);
  fwd_fft_ref->set_input (unpacked_ref);
  fwd_fft_ref->set_output (channelised_ref);
  fwd_fft_ref->set_nfft (nfft);
  fwd_fft_ref->set_verbose (verbose);

  // cleaned data
  cleaned = new spip::ContainerCUDADevice();

  // RFI Filtering operation
  filter = new spip::AdaptiveFilterCUDA();
  filter->set_input (channelised);
  filter->set_input_ref (channelised_ref);
  filter->set_output (cleaned);
  filter->set_verbose (verbose);

  // output of BWD fft
  d_output = new spip::ContainerCUDADevice ();

  if (verbose)
    cerr << "spip::AdaptiveFilterPipeline::configure_cuda allocating Backward FFT" << endl;
  // backward fft operation
  bwd_fft = new spip::BackwardFFTCUDA(stream);
  bwd_fft->set_input (cleaned);
  bwd_fft->set_output (d_output);
  bwd_fft->set_nfft (nfft);
  bwd_fft->set_verbose (verbose);

  if (verbose)
    cerr << "spip::AdaptiveFilterPipeline::configure_cuda allocating output Ring Buffer" << endl;
  // coarse output channels
  output = new spip::ContainerRingWrite (out_db);
  output->register_buffers();

  // transfer device to host
  if (verbose)
    cerr << "spip::AdaptiveFilterPipeline::configure_cuda allocating CUDA to RAM Transfer" << endl;
  cuda_to_ram = new spip::CUDAtoRAMTransfer (stream);
  cuda_to_ram->set_input (d_output);
  cuda_to_ram->set_output (output);
  cuda_to_ram->set_verbose (verbose);
}

#endif

//! process meta-data through the pipeline, performing all resource allocation
void spip::AdaptiveFilterPipeline::open ()
{
  if (verbose)
    cerr << "spip::AdaptiveFilterPipeline::open()" << endl;

  if (verbose)
    cerr << "spip::AdaptiveFilterPipeline::open input->read_header()" << endl;
  // read from the input
  input->process_header();
  input_ref->process_header();

#ifdef HAVE_CUDA
  if (device >= 0)
  {
    if (verbose)
      cerr << "spip::AdaptiveFilterPipeline::open ram_to_cuda->configure()" << endl;
    ram_to_cuda->configure(spip::Ordering::SFPT);
    ram_to_cuda_ref->configure(spip::Ordering::SFPT);
  }
#endif
  
  // configure the unpacker
  if (verbose)
    cerr << "spip::AdaptiveFilterPipeline::open unpack_float->configure()" << endl;
  unpack_float->configure(spip::Ordering::SFPT);
  unpack_float_ref->configure(spip::Ordering::SFPT);

  // configure the forward FFT
  if (verbose)
    cerr << "spip::AdaptiveFilterPipeline::open fwd_fft->configure()" << endl;
  fwd_fft->configure(spip::Ordering::SFPT);
  fwd_fft_ref->configure(spip::Ordering::SFPT);

  if (verbose)
    cerr << "spip::AdaptiveFilterPipeline::open filter->configure()" << endl;
  filter->configure(spip::Ordering::SFPT);

  // configure the backward FFTs
  if (verbose)
    cerr << "spip::AdaptiveFilterPipeline::open bwd_fft->configure()" << endl;
  bwd_fft->configure(spip::Ordering::SFPT);

#ifdef HAVE_CUDA
  if (device >= 0)
  {
    if (verbose)
      cerr << "spip::AdaptiveFilterPipeline::open cuda_to_ram->configure()" << endl;
    cuda_to_ram->configure(spip::Ordering::SFPT);
  }
#endif

  // write the output header
  if (verbose)
    cerr << "spip::AdaptiveFilterPipeline::open output->write_header()" << endl;
  output->process_header();
}

//! close the input and output data blocks
void spip::AdaptiveFilterPipeline::close ()
{
  if (verbose)
    cerr << "spip::AdaptiveFilterPipeline::close()" << endl;

  if (out_db->is_block_open())
  {
    if (verbose)
      cerr << "spip::AdaptiveFilterPipeline::close out_db->close_block(" << out_db->get_data_bufsz() << ")" << endl;
    out_db->close_block (out_db->get_data_bufsz());
  }

  if (in_db->is_block_open())
  {
    if (verbose)
      cerr << "spip::AdaptiveFilterPipeline::close in_db->close_block(" << in_db->get_data_bufsz() << ")" << endl;
    in_db->close_block (in_db->get_data_bufsz());
  }

  if (ref_db->is_block_open())
  {
    if (verbose)
      cerr << "spip::AdaptiveFilterPipeline::close ref_db->close_block(" << ref_db->get_data_bufsz() << ")" << endl;
    ref_db->close_block (ref_db->get_data_bufsz());
  }


  // close the data blocks, ending the observation
  if (verbose)
    cerr << "spip::AdaptiveFilterPipeline::close in_db->close()" << endl;
  in_db->close();

  if (verbose)
    cerr << "spip::AdaptiveFilterPipeline::close ref_db->close()" << endl;
  ref_db->close();

  if (verbose)
    cerr << "spip::AdaptiveFilterPipeline::close out_db->close()" << endl;
  out_db->close();
}

// process blocks of input data until the end of the data stream
bool spip::AdaptiveFilterPipeline::process ()
{
  if (verbose)
    cerr << "spip::AdaptiveFilterPipeline::process ()" << endl;

  bool keep_processing = true;

  // commence observation on output data block
  if (verbose)
    cerr << "spip::AdaptiveFilterPipeline::out_db->open()" << endl;
  out_db->open();

  uint64_t input_bufsz = in_db->get_data_bufsz();
  uint64_t ref_bufsz = ref_db->get_data_bufsz();
  uint64_t nbytes_input, nbytes_ref;

  while (keep_processing)
  {
    // read a block of input data
    if (verbose)
      cerr << "spip::AdaptiveFilterPipeline::process input->open_block()" << endl;
    nbytes_input = input->open_block();
    if (verbose)
      cerr << "spip::AdaptiveFilterPipeline::process input_ref->open_block()" << endl;
    nbytes_ref = input_ref->open_block();

    if (nbytes_input < input_bufsz || nbytes_ref < ref_bufsz)
      keep_processing = false;

    // only process full blocks of data
    if (keep_processing)
    {
      // open a block of output data
      if (verbose)
        cerr << "spip::AdaptiveFilterPipeline::process output->open_block()" << endl;
      output->open_block();

  #ifdef HAVE_CUDA
      if (device >= 0)
      {
        ram_to_cuda->prepare();
        ram_to_cuda->transformation();
        ram_to_cuda_ref->prepare();
        ram_to_cuda_ref->transformation();
      }
  #endif

      if (verbose)
        cerr << "spip::AdaptiveFilterPipeline::process unpack_float->transformation()" << endl;
      unpack_float->prepare();
      unpack_float->transformation();
      if (verbose)
        cerr << "spip::AdaptiveFilterPipeline::process unpack_float_ref->transformation()" << endl;
      unpack_float_ref->prepare();
      unpack_float_ref->transformation();

      // perform Forward FFT operation
      if (verbose)
        cerr << "spip::AdaptiveFilterPipeline::process fwd_fft->transformation()" << endl;
      fwd_fft->prepare();
      fwd_fft->transformation ();
      if (verbose)
        cerr << "spip::AdaptiveFilterPipeline::process fwd_fft_ref->transformation()" << endl;
      fwd_fft_ref->prepare();
      fwd_fft_ref->transformation ();

      if (verbose)
        cerr << "spip::AdaptiveFilterPipeline::process filter->transformation()" << endl;
      filter->prepare();
      filter->transformation();

      // perform the Backward FFT operation
      if (verbose)
        cerr << "spip::AdaptiveFilterPipeline::process bwd_fft->transformation()" << endl;
      bwd_fft->prepare();
      bwd_fft->transformation ();

#ifdef HAVE_CUDA
      if (device >= 0)
      {
        cuda_to_ram->prepare();
        cuda_to_ram->transformation();
      }
#endif
      if (verbose)
        cerr << "spip::AdaptiveFilterPipeline::process output->close_block()" << endl;
      output->close_block();
    }
    if (verbose)
      cerr << "spip::AdaptiveFilterPipeline::process input->close_block()" << endl;
    input->close_block();
    if (verbose)
      cerr << "spip::AdaptiveFilterPipeline::process input_ref->close_block()" << endl;
    input_ref->close_block();
  }

  // close the data blocks
  close();

  return true;
}
