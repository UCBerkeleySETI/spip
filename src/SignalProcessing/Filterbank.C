/***************************************************************************
 *
 *   Copyright (C) 2016 Andrew Jameson
 *   Licensed under the Academic Free License version 2.1
 *
 ***************************************************************************/

#include "spip/Filterbank.h"

#include <signal.h>
#include <unistd.h>
#include <cstring>
#include <iostream>
#include <stdexcept>
#include <new>

//#define _DEBUG

using namespace std;

spip::Filterbank::Filterbank (const char * in_key_string, const char * out_key_string)
{
  nfft_fwd = -1;
  nfft_bwd = -1;

  in_db  = new DataBlockRead (in_key_string);
  out_db = new DataBlockWrite (out_key_string);

  in_db->connect();
  in_db->lock();

  out_db->connect();
  out_db->lock();

  verbose = false;
}

spip::Filterbank::~Filterbank()
{
  in_db->unlock();
  in_db->disconnect();
  delete in_db;

  out_db->unlock();
  out_db->disconnect();
  delete out_db;
}

void spip::Filterbank::set_channelisation (int freq_res, int nchan_out)
{
  if (verbose)
    cerr << "spip::Filterbank::set_channelisation freq_res=" << freq_res 
         << " nchan_out=" << nchan_out << endl;
  nfft_fwd = nchan_out * freq_res;
  nfft_bwd = freq_res;
}

//! build the pipeline containers and transforms
void spip::Filterbank::configure ()
{
  if (verbose)
    cerr << "spip::Filterbank::configure ()" << endl;
#ifdef HAVE_CUDA
  if (device >= 0)
    return configure_cuda();
#endif
  
  if (verbose)
    cerr << "spip::Filterbank::configure creating input" << endl;
  // input container, reads header 
  input = new spip::ContainerRingRead (in_db);

  if (verbose)
    cerr << "spip::Filterbank::configure unpacked container" << endl;
  // unpacked container
  unpacked = new spip::ContainerRAM ();
  
  if (verbose)
    cerr << "spip::Filterbank::configure allocating UnpackFloat" << endl;
  // unpack to float operation
  unpack_float = new spip::UnpackFloatRAM();
  unpack_float->set_input (input);
  unpack_float->set_output (unpacked);
  unpack_float->set_verbose (verbose);

  // fine channels
  if (verbose)
    cerr << "spip::Filterbank::configure allocating channelised container" << endl;
  channelised = new spip::ContainerRAM ();
  
  if (verbose)
    cerr << "spip::Filterbank::configure allocating Forward FFT" << endl;
  // forward FFT operation
  fwd_fft = new spip::ForwardFFTFFTW();
  fwd_fft->set_input (unpacked);
  fwd_fft->set_output (channelised);
  fwd_fft->set_nfft (nfft_fwd);
  fwd_fft->set_verbose (verbose);

  if (verbose)
    cerr << "spip::Filterbank::configure allocating output Ring Buffer" << endl;
  // coarse output channels
  output = new spip::ContainerRingWrite (out_db);
  
  if (verbose)
    cerr << "spip::Filterbank::configure allocating Backward FFT" << endl;
  // backward fft operation
  bwd_fft = new spip::BatchedBackwardFFTFFTW();
  bwd_fft->set_input (channelised);
  bwd_fft->set_output (output);
  bwd_fft->set_nfft (nfft_bwd);
  bwd_fft->set_verbose (verbose);
}

#ifdef HAVE_CUDA

void spip::Filterbank::set_device (int _device)
{
  int device_count = 0;
  cudaError_t err = cudaGetDeviceCount(&device_count);
  if (err != cudaSuccess)
    throw runtime_error ("spip::Filterbank::set_device cudaGetDeviceCount failed");

  device = _device;
  if (device >= device_count)
    throw runtime_error ("Specified CUDA device not available");

  // TODO error checking
  err = cudaSetDevice (device);
  if (err != cudaSuccess)
    throw runtime_error ("spip::Filterbank::set_device cudaSetDevice failed");

  err = cudaStreamCreate(&stream);
   if (err != cudaSuccess)
    throw runtime_error ("spip::Filterbank::set_device cudaStreamCreate failed");
}
  
//! build the pipeline containers and transforms
void spip::Filterbank::configure_cuda ()
{
  if (verbose)
    cerr << "spip::Filterbank::configure_cuda creating input" << endl;
  // input container, reads header 
  input = new spip::ContainerRingRead (in_db);
  input->register_buffers();

  // transfer host to device
  d_input = new spip::ContainerCUDADevice ();
  if (verbose)
    cerr << "spip::Filterbank::configure_cuda allocating RAM to CUDA Transfer" << endl;
  ram_to_cuda = new spip::RAMtoCUDATransfer (stream);
  ram_to_cuda->set_input (input);
  ram_to_cuda->set_output (d_input); 
  ram_to_cuda->set_verbose (verbose);

  if (verbose)
    cerr << "spip::Filterbank::configure_cuda unpacked container" << endl;
  // unpacked container
  unpacked = new spip::ContainerCUDADevice ();
  if (verbose)
    cerr << "spip::Filterbank::configure_cuda allocating UnpackFloat" << endl;

  // unpack to float operation
  unpack_float = new spip::UnpackFloatCUDA(stream);
  unpack_float->set_input (d_input);
  unpack_float->set_output (unpacked);
  unpack_float->set_verbose (verbose);

  // fine channels
  if (verbose)
    cerr << "spip::Filterbank::configure_cuda allocating channelised container" << endl;
  channelised = new spip::ContainerCUDADevice ();

  if (verbose)
    cerr << "spip::Filterbank::configure_cuda allocating Forward FFT" << endl;
  // forward FFT operation
  fwd_fft = new spip::ForwardFFTCUDA(stream);
  fwd_fft->set_input (unpacked);
  fwd_fft->set_output (channelised);
  fwd_fft->set_nfft (nfft_fwd);
  fwd_fft->set_verbose (verbose);

  // output of BWD fft
  d_output = new spip::ContainerCUDADevice ();

  if (verbose)
    cerr << "spip::Filterbank::configure_cuda allocating Backward FFT" << endl;
  // backward fft operation
  bwd_fft = new spip::BatchedBackwardFFTCUDA(stream);
  bwd_fft->set_input (channelised);
  bwd_fft->set_output (d_output);
  bwd_fft->set_nfft (nfft_bwd);
  bwd_fft->set_verbose (verbose);

  if (verbose)
    cerr << "spip::Filterbank::configure_cuda allocating output Ring Buffer" << endl;
  // coarse output channels
  output = new spip::ContainerRingWrite (out_db);
  output->register_buffers();

  // transfer device to host
  if (verbose)
    cerr << "spip::Filterbank::configure_cuda allocating CUDA to RAM Transfer" << endl;
  cuda_to_ram = new spip::CUDAtoRAMTransfer (stream);
  cuda_to_ram->set_input (d_output);
  cuda_to_ram->set_output (output);
  cuda_to_ram->set_verbose (verbose);
}

#endif

//! process meta-data through the pipeline, performing all resource allocation
void spip::Filterbank::open ()
{
  if (verbose)
    cerr << "spip::Filterbank::open()" << endl;

  if (verbose)
    cerr << "spip::Filterbank::open input->read_header()" << endl;
  // read from the input
  input->process_header();

#ifdef HAVE_CUDA
  if (device >= 0)
  {
    if (verbose)
      cerr << "spip::Filterbank::open ram_to_cuda->configure()" << endl;
    ram_to_cuda->configure();
  }
#endif
  
  // configure the unpacker
  if (verbose)
    cerr << "spip::Filterbank::open unpack_float->configure()" << endl;
  unpack_float->configure();

  // configure the forward FFT
  if (verbose)
    cerr << "spip::Filterbank::open fwd_fft->configure()" << endl;
  fwd_fft->configure();

  // configure the backward FFTs
  if (verbose)
    cerr << "spip::Filterbank::open bwd_fft->configure()" << endl;
  bwd_fft->configure();

#ifdef HAVE_CUDA
  if (device >= 0)
  {
    if (verbose)
      cerr << "spip::Filterbank::open cuda_to_ram->configure()" << endl;
    cuda_to_ram->configure();
  }
#endif

  // write the output header
  if (verbose)
    cerr << "spip::Filterbank::open output->write_header()" << endl;
  output->process_header();
}

//! close the input and output data blocks
void spip::Filterbank::close ()
{
  if (verbose)
    cerr << "spip::Filterbank::close()" << endl;

  if (out_db->is_block_open())
  {
    if (verbose)
      cerr << "spip::Filterbank::close out_db->close_block(" << out_db->get_data_bufsz() << ")" << endl;
    out_db->close_block (out_db->get_data_bufsz());
  }

  if (in_db->is_block_open())
  {
    if (verbose)
      cerr << "spip::Filterbank::close in_db->close_block(" << in_db->get_data_bufsz() << ")" << endl;
    in_db->close_block (in_db->get_data_bufsz());
  }

  // close the data blocks, ending the observation
  if (verbose)
    cerr << "spip::Filterbank::close in_db->close()" << endl;
  in_db->close();

  if (verbose)
    cerr << "spip::Filterbank::close out_db->close()" << endl;
  out_db->close();
}

// process blocks of input data until the end of the data stream
bool spip::Filterbank::process ()
{
  if (verbose)
    cerr << "spip::Filterbank::process ()" << endl;

  bool keep_processing = true;

  // commence observation on output data block
  if (verbose)
    cerr << "spip::Filterbank::out_db->open()" << endl;
  out_db->open();

  uint64_t input_bufsz = in_db->get_data_bufsz();
  uint64_t nbytes;

  while (keep_processing)
  {
    // read a block of input data
    if (verbose)
      cerr << "spip::Filterbank::process input->open_block()" << endl;
    nbytes = input->open_block();

    if (nbytes < input_bufsz)
      keep_processing = false;

    // open a block of output data
    if (verbose)
      cerr << "spip::Filterbank::process output->open_block()" << endl;
    output->open_block();

#ifdef HAVE_CUDA
  if (device >= 0)
  {
    ram_to_cuda->prepare();
    ram_to_cuda->transformation();
  }
#endif

    if (verbose)
      cerr << "spip::Filterbank::process unpack_float->transformation()" << endl;
    unpack_float->prepare();
    unpack_float->transformation();

    // perform Forward FFT operation
    if (verbose)
      cerr << "spip::Filterbank::process fwd_fft->transformation()" << endl;
    fwd_fft->prepare();
    fwd_fft->transformation ();

    // perform the Backward FFT operation
    if (verbose)
      cerr << "spip::Filterbank::process bwd_fft->transformation()" << endl;
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
      cerr << "spip::Filterbank::process input->close_block()" << endl;
    input->close_block();

    if (verbose)
      cerr << "spip::Filterbank::process output->close_block()" << endl;
    output->close_block();
  }

  // close the data blocks
  close();

  return true;
}
