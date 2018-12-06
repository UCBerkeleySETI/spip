/***************************************************************************
 *
 *   Copyright (C) 2018 Andrew Jameson
 *   Licensed under the Academic Free License version 2.1
 *
 ***************************************************************************/

#include "spip/ContinuumPipeline.h"

#include <signal.h>
#include <unistd.h>
#include <cstring>
#include <iostream>
#include <stdexcept>
#include <new>

//#define _DEBUG

using namespace std;

spip::ContinuumPipeline::ContinuumPipeline (const char * in_key_string, const char * out_dir_string)
{
  output_state = spip::Signal::Coherence;
  channel_oversampling = 1;
  tdec = 1;
  fdec = 1;
  tsubint = 10;

  nchan_out = -1;
  nfft = -1;

#ifdef HAVE_CUDA
  device = -1;
  input_ring_ram = true;
#endif

  in_db  = new DataBlockRead (in_key_string);
  out_dir = string(out_dir_string);

  in_db->connect();
  in_db->lock();

  verbose = false;
  unpack = true;
}

spip::ContinuumPipeline::~ContinuumPipeline()
{
  in_db->unlock();
  in_db->disconnect();
  delete in_db;
}

void spip::ContinuumPipeline::set_channelisation (int _nchan)
{
  nchan_out = _nchan;
  nfft = nchan_out * channel_oversampling;
  if (verbose) 
    cerr << "spip::ContinuumPipeline::set_channelisation nchan_out=" << nchan_out << " nfft=" << nfft << endl;
}

void spip::ContinuumPipeline::set_channel_oversampling (int _factor)
{
  if (verbose)
    cerr << "spip::ContinuumPipeline::set_channel_oversampling factor=" << _factor << endl;
  channel_oversampling = _factor;
  nfft = nchan_out * channel_oversampling;
}

void spip::ContinuumPipeline::set_decimation (int _tdec)
{
  tdec = _tdec;
  fdec = channel_oversampling;
}

void spip::ContinuumPipeline::set_tsubint (float _tsubint)
{
  tsubint = _tsubint;
}

void spip::ContinuumPipeline::set_output_state (spip::Signal::State _state)
{
  output_state = _state;
}

//! build the pipeline containers and transforms
void spip::ContinuumPipeline::configure (spip::UnpackFloat * unpacker)
{
  if (verbose)
    cerr << "spip::ContinuumPipeline::configure ()" << endl;
#ifdef HAVE_CUDA
  if (device >= 0)
    return configure_cuda (unpacker);
#endif
  
  if (verbose)
    cerr << "spip::ContinuumPipeline::configure creating input" << endl;
  // input containers, reads header 
  input = new spip::ContainerRingRead (in_db);

  if (verbose)
    cerr << "spip::ContinuumPipeline::configure allocating reblocked container" << endl;
  reblocked = new spip::ContainerRAM ();

  if (verbose)
    cerr << "spip::ContinuumPipeline::configure allocating RAMtoRAMTransfer" << endl;
  ram_to_ram = new spip::RAMtoRAMTransfer();
  ram_to_ram->set_input (input);
  ram_to_ram->set_output (reblocked);
  ram_to_ram->set_verbose (verbose);

  if (verbose)
    cerr << "spip::ContinuumPipeline::configure unpacked container" << endl;
  // unpacked containers
  unpacked = new spip::ContainerRAM ();
  
  if (verbose)
    cerr << "spip::ContinuumPipeline::configure allocating UnpackFloat" << endl;
  // unpack to float operation
  unpack_float = unpacker;
  unpack_float->set_input (reblocked);
  unpack_float->set_output (unpacked);
  unpack_float->set_verbose (verbose);

  // fine channels
  if (verbose)
    cerr << "spip::ContinuumPipeline::configure allocating channelised container" << endl;
  channelised = new spip::ContainerRAM ();

  if (verbose)
    cerr << "spip::ContinuumPipeline::configure allocating Forward FFT" << endl;
  // forward FFT operation
  fwd_fft = new spip::ForwardFFTFFTW();
  fwd_fft->set_input (unpacked);
  fwd_fft->set_output (channelised);
  fwd_fft->set_nfft (nfft);
  fwd_fft->set_verbose (verbose);

  // detected data
  detected = new spip::ContainerRAM ();

  if (verbose)
    cerr << "spip::ContinuumPipeline::configure allocating Detection" << endl;
  // Detector
  if (output_state == spip::Signal::Intensity)
  {
    detector = new spip::DetectionSquareLawRAM();
    cerr << "output_state == spip::Signal::Intensity" << endl;
  }
  else if (output_state == spip::Signal::PPQQ)
  {
    detector = new spip::DetectionSquareLawRAM();
    cerr << "output_state == spip::Signal::PPQQ" << endl;
  }
  else
  {
    detector = new spip::DetectionPolarimetryRAM();
    cerr << "output_state == other" << endl;
  }
  detector->set_output_state (output_state);
  detector->set_input (channelised);
  detector->set_output (detected);
  detector->set_verbose (verbose);

  // integrated data
  //integrated = new spip::ContainerRAM ();

  // lower sideband data
  if (verbose)
    cerr << "spip::ContinuumPipeline::configure allocating output File Writer" << endl;
  output = new spip::ContainerFileWrite (out_dir);
  output->set_file_length_seconds (tsubint);

  // Temporal and spectral integrator 
  if (verbose)
    cerr << "spip::ContinuumPipeline::configure integrator->set_decimation(" << tdec << ", 1,  " << fdec << ", 1)" << endl;
  integrator = new spip::IntegrationRAM();
  integrator->set_decimation (tdec, 1, fdec, 1);
  integrator->set_input (detected);
  integrator->set_output (output);
  integrator->set_verbose (verbose);
}

#ifdef HAVE_CUDA

void spip::ContinuumPipeline::set_device (int _device)
{
  int device_count = 0;
  cudaError_t err = cudaGetDeviceCount(&device_count);
  if (err != cudaSuccess)
    throw runtime_error ("spip::ContinuumPipeline::set_device cudaGetDeviceCount failed");

  device = _device;
  if (device >= device_count)
    throw runtime_error ("Specified CUDA device not available");

  // TODO error checking
  err = cudaSetDevice (device);
  if (err != cudaSuccess)
    throw runtime_error ("spip::ContinuumPipeline::set_device cudaSetDevice failed");

  err = cudaStreamCreate(&stream);
   if (err != cudaSuccess)
    throw runtime_error ("spip::ContinuumPipeline::set_device cudaStreamCreate failed");
}
  
//! build the pipeline containers and transforms
void spip::ContinuumPipeline::configure_cuda (spip::UnpackFloat * unpacker)
{
  // input container, reads header 
  if (verbose)
    cerr << "spip::ContinuumPipeline::configure_cuda creating input" << endl;
  // if input data block resides on the host
  if (in_db->get_device() == -1)
  {
    input = new spip::ContainerRingRead (in_db);
    input->register_buffers();
  }
  else
  {
    d_input = new spip::ContainerRingReadCUDA (in_db);
    unpack = false;
    input_ring_ram = false;
  }

  // transfer host to device
  reblocked = new spip::ContainerCUDADevice ();

  if (input_ring_ram)
  {
    if (verbose)
      cerr << "spip::ContinuumPipeline::configure_cuda allocating ram_to_cuda" << endl;
    ram_to_cuda = new spip::RAMtoCUDATransfer (stream);
    ram_to_cuda->set_input (input);
    ram_to_cuda->set_output (dynamic_cast<spip::ContainerCUDADevice*>(reblocked));
    ram_to_cuda->set_verbose (verbose);
  }
  else
  {
    if (verbose)
      cerr << "spip::ContinuumPipeline::configure_cuda allocating cuda_to_cuda" << endl;
    cuda_to_cuda = new spip::CUDARingtoCUDATransfer (stream);
    cuda_to_cuda->set_input (d_input);
    cuda_to_cuda->set_output (dynamic_cast<spip::ContainerCUDADevice*>(reblocked));
    cuda_to_cuda->set_verbose (verbose);
  }

  if (unpack)
  {
    if (verbose)
      cerr << "spip::ContinuumPipeline::configure_cuda unpacked container" << endl;
    // unpacked container
    unpacked = new spip::ContainerCUDADevice ();
    if (verbose)
      cerr << "spip::ContinuumPipeline::configure_cuda allocating UnpackFloat" << endl;

    // unpack to float operation
    unpack_float = unpacker;
    unpack_float->set_input (reblocked);
    unpack_float->set_output (unpacked);
    unpack_float->set_verbose (verbose);

    // ensure the cuda Stream is set
    spip::UnpackFloatCUDA * tmp = dynamic_cast<spip::UnpackFloatCUDA *>(unpacker);
    tmp->set_stream (stream);
  }
  
  // fine channels
  if (verbose)
    cerr << "spip::ContinuumPipeline::configure_cuda allocating channelised container" << endl;
  channelised = new spip::ContainerCUDADevice ();

  if (verbose)
    cerr << "spip::ContinuumPipeline::configure_cuda allocating Forward FFT" << endl;
  // forward FFT operation
  fwd_fft = new spip::ForwardFFTCUDA(stream);
  if (unpack)
    fwd_fft->set_input (unpacked);
  else
    fwd_fft->set_input (reblocked);
  fwd_fft->set_output (channelised);
  fwd_fft->set_nfft (nfft);
  fwd_fft->set_verbose (verbose);

  // detected data
  detected = new spip::ContainerCUDADevice();

  // coherence detector
  if ((output_state == spip::Signal::Intensity) || (output_state == spip::Signal::PPQQ))
    detector = new spip::DetectionSquareLawCUDA(stream);
  else
    detector = new spip::DetectionPolarimetryCUDA(stream);
  detector->set_output_state (output_state);
  detector->set_input (channelised);
  detector->set_output (detected);
  detector->set_verbose (verbose);

  // integrated data
  integrated = new spip::ContainerCUDADevice();

  // lower sideband data
  d_output = new spip::ContainerCUDADevice();

  // time and frequency integrator
  integrator = new spip::IntegrationCUDA(stream);
  integrator->set_decimation (tdec, 1, fdec, 1);
  integrator->set_input (detected);
  integrator->set_output (d_output);
  integrator->set_verbose (verbose);

  if (verbose)
    cerr << "spip::ContinuumPipeline::configure_cuda allocating output File Writer tsubint=" << tsubint << endl;
  // coarse output channels
  output = new spip::ContainerFileWrite(out_dir);
  output->register_buffer();
  output->set_file_length_seconds (tsubint);

  // transfer device to host
  if (verbose)
    cerr << "spip::ContinuumPipeline::configure_cuda allocating CUDA to RAM Transfer" << endl;
  cuda_to_ram = new spip::CUDAtoRAMTransfer (stream);
  cuda_to_ram->set_input (d_output);
  cuda_to_ram->set_output (output);
  cuda_to_ram->set_verbose (verbose);
}

#endif

//! process meta-data through the pipeline, performing all resource allocation
void spip::ContinuumPipeline::open ()
{
  if (verbose)
    cerr << "spip::ContinuumPipeline::open()" << endl;

  if (verbose)
    cerr << "spip::ContinuumPipeline::open input->read_header()" << endl;

  // read from the input
#ifdef HAVE_CUDA
  if (in_db->get_device() != -1)
  {
    input_ring = dynamic_cast<spip::ContainerRing*>(d_input);
  }
  else
#endif
  {
    input_ring = dynamic_cast<spip::ContainerRing*>(input);
  }

  input_ring->process_header();

  // now determine the required reblocking
  uint64_t input_block_size = input_ring->get_size();
  uint64_t bytes_required = uint64_t(input_ring->calculate_nbits_per_sample()) * nfft / 8;
  if (verbose)
    cerr << "spip::ContinuumPipeline::open input_ring->calculate_nbits_per_sample()=" 
         << input_ring->calculate_nbits_per_sample() << " nfft=" << nfft << endl;
  if (bytes_required > input_block_size)
  {
    reblock_factor = bytes_required / input_block_size;
    if (bytes_required % input_block_size)
    {
      throw Error (InvalidState, "spip::ContinuumPipeline::open", 
                   "bytes_required [%lu] % input_block_size [%lu] != 0\n", 
                    bytes_required, input_block_size);
    }
  }
  else
    reblock_factor = 1;

  if (verbose)
    cerr << "spip::ContinuumPipeline::open input_block_size=" << input_block_size 
         << " bytes_required=" << bytes_required << " reblock_factor=" << reblock_factor << endl;
   
#ifdef HAVE_CUDA
  if (device >= 0)
  {
    if (input_ring_ram)
    {
      if (verbose)
        cerr << "spip::ContinuumPipeline::open ram_to_cuda->configure(SFPT)" << endl;
      ram_to_cuda->set_output_reblock (reblock_factor);
      ram_to_cuda->configure(spip::Ordering::SFPT);
    }
    else
    {
      if (verbose)
        cerr << "spip::ContinuumPipeline::open cuda_to_cuda->set_output_reblock(" << reblock_factor << ")" << endl;
      cuda_to_cuda->set_output_reblock (reblock_factor);
      if (verbose)
        cerr << "spip::ContinuumPipeline::open cuda_to_cuda->configure(SFPT)" << endl;
      cuda_to_cuda->configure(spip::Ordering::SFPT);
    }
  }
  else
#endif
  {
    ram_to_ram->set_output_reblock (reblock_factor);
    ram_to_ram->configure (spip::Ordering::SFPT);
  }
  
  if (unpack)
  {
    // configure the unpacker
    if (verbose)
      cerr << "spip::ContinuumPipeline::open unpack_float->configure(SFPT)" << endl;
    unpack_float->configure(spip::Ordering::SFPT);
  }

  // configure the forward FFT
  if (verbose)
    cerr << "spip::ContinuumPipeline::open fwd_fft->configure(TSPF)" << endl;
  fwd_fft->configure(spip::Ordering::TSPF);

  if (verbose)
    cerr << "spip::ContinuumPipeline::open detector->configure(TSPF)" << endl;
  detector->configure(spip::Ordering::TSPF);

  if (verbose)
    cerr << "spip::ContinuumPipeline::open integrator->configure(TSPF)" << endl;
  integrator->configure(spip::Ordering::TSPF);

#ifdef HAVE_CUDA
  if (device >= 0)
  {
    if (verbose)
      cerr << "spip::ContinuumPipeline::open cuda_to_ram->configure()" << endl;
    cuda_to_ram->configure(spip::Ordering::TSPF);
  }
#endif

  // write the output header
  if (verbose)
    cerr << "spip::ContinuumPipeline::open output->process_header()" << endl;
  output->process_header();
}



//! close the input and output data blocks
void spip::ContinuumPipeline::close ()
{
  if (verbose)
    cerr << "spip::ContinuumPipeline::close()" << endl;

  if (in_db->is_block_open())
  {
    if (verbose)
      cerr << "spip::ContinuumPipeline::close in_db->close_block(" << in_db->get_data_bufsz() << ")" << endl;
    in_db->close_block (in_db->get_data_bufsz());
  }

  // close the input data block
  if (verbose)
    cerr << "spip::ContinuumPipeline::close in_db->close()" << endl;
  in_db->close();
}

// process blocks of input data until the end of the data stream
bool spip::ContinuumPipeline::process ()
{
  if (verbose)
    cerr << "spip::ContinuumPipeline::process ()" << endl;

  bool keep_processing = true;

  uint64_t input_bufsz = in_db->get_data_bufsz();
  uint64_t nbytes_input;

  while (keep_processing)
  {
    unsigned blocks_read = 0;
    while (keep_processing && blocks_read < reblock_factor)
    {
      // read a block of input data
      if (verbose)
        cerr << "spip::ContinuumPipeline::process input_ring->open_block()" << endl;
      nbytes_input = input_ring->open_block();
      if (verbose)
        cerr << "spip::ContinuumPipeline::process input block contains " << nbytes_input << " bytes" << endl;

      if (nbytes_input < input_bufsz)
      {
        keep_processing = false;
      }

      // only process full blocks of data
      if (keep_processing)
      {
#ifdef HAVE_CUDA
        if (device >= 0)
        {
          if (input_ring_ram)
          {
            ram_to_cuda->prepare();
            ram_to_cuda->transformation();
          }
          else
          {
            cuda_to_cuda->prepare();
            cuda_to_cuda->transformation();
          }
        }
        else
#endif
        {
          // inefficient for nblock == 1, but its not the performant pipeline
          ram_to_ram->prepare();
          ram_to_ram->transformation();
        }
      }

      blocks_read++;

      if (verbose)
        cerr << "spip::ContinuumPipeline::process input_ring->close_block()" << endl;
      input_ring->close_block();
    }

    if (unpack)
    {
      if (verbose)
        cerr << "spip::ContinuumPipeline::process unpack_float->transformation()" << endl;
      unpack_float->prepare();
      unpack_float->transformation();
    }
  
    if (keep_processing)
    {
      // perform Forward FFT operation
      if (verbose)
        cerr << "spip::ContinuumPipeline::process fwd_fft->transformation()" << endl;
      fwd_fft->prepare();
      fwd_fft->transformation ();

      if (verbose)
        cerr << "spip::ContinuumPipeline::process detector->transformation()" << endl;
      detector->prepare();
      detector->transformation();

      if (verbose)
        cerr << "spip::ContinuumPipeline::process integrator->transformation()" << endl;
      integrator->prepare();
      integrator->transformation();


#ifdef HAVE_CUDA
      if (device >= 0)
      {
        if (verbose)
          cerr << "spip::ContinuumPipeline::process cuda_to_ram->transformation()" << endl;
        cuda_to_ram->prepare();
        cuda_to_ram->transformation();
      }
#endif

      if (verbose)
        cerr << "spip::ContinuumPipeline::process output->write()" << endl;
      output->write ();
    }

  }

  // close the data blocks
  close();

  return true;
}
