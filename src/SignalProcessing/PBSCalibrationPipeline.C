/***************************************************************************
 *
 *   Copyright (C) 2018 Andrew Jameson
 *   Licensed under the Academic Free License version 2.1
 *
 ***************************************************************************/

#include "spip/PBSCalibrationPipeline.h"

#include <signal.h>
#include <unistd.h>
#include <cstring>
#include <iostream>
#include <stdexcept>
#include <new>

//#define _DEBUG

using namespace std;

spip::PBSCalibrationPipeline::PBSCalibrationPipeline (const char * in_key_string, const char * out_key_string)
{
  output_state = spip::Signal::Coherence;
  dat_dec = 1;
  dat_offset = 0;
  nbin = 1;

#ifdef HAVE_CUDA
  device = -1;
#endif

  in_db  = new DataBlockRead (in_key_string);
  out_db = new DataBlockWrite (out_key_string);

  in_db->connect();
  in_db->lock();

  out_db->connect();
  out_db->lock();

  verbose = false;
}

spip::PBSCalibrationPipeline::~PBSCalibrationPipeline()
{
  in_db->unlock();
  in_db->disconnect();
  delete in_db;

  out_db->unlock();
  out_db->disconnect();
  delete out_db;
}

void spip::PBSCalibrationPipeline::set_periodicity (unsigned _nbin, uint64_t _dat_dec)
{
  if (verbose) 
    cerr << "spip::PBSCalibrationPipeline::set_periodicity nbin=" << _nbin << " dat_dec=" << _dat_dec << endl;
  nbin = _nbin;
  dat_offset = 0;
  dat_dec = _dat_dec;
}

void spip::PBSCalibrationPipeline::set_output_state (spip::Signal::State _state)
{
  output_state = _state;
}

//! build the pipeline containers and transforms
void spip::PBSCalibrationPipeline::configure (spip::UnpackFloat * unpacker)
{
  if (verbose)
    cerr << "spip::PBSCalibrationPipeline::configure ()" << endl;
#ifdef HAVE_CUDA
  if (device >= 0)
    return configure_cuda(unpacker);
#endif
  
  if (verbose)
    cerr << "spip::PBSCalibrationPipeline::configure creating input" << endl;
  // input containers, reads header 
  input = new spip::ContainerRingRead (in_db);

  if (verbose)
    cerr << "spip::PBSCalibrationPipeline::configure unpacked container" << endl;
  unpacked = new spip::ContainerRAM ();
  
  if (verbose)
    cerr << "spip::PBSCalibrationPipeline::configure allocating UnpackFloat" << endl;
  // unpack to float operation
  unpack_float = unpacker;
  unpack_float->set_input (input);
  unpack_float->set_output (unpacked);
  unpack_float->set_verbose (verbose);

  if (verbose)
    cerr << "spip::PBSCalibrationPipeline::configure allocating output Ring Buffer" << endl;
  output = new spip::ContainerBufferedRingWrite (out_db);

  if (verbose)
    cerr << "spip::PBSCalibrationPipeline::configure allocating SampleFold" << endl;
  // sample fold operation
  sample_fold = new spip::SampleFoldRAM();
  sample_fold->set_input (unpacked);
  sample_fold->set_output (output);
  cerr << "spip::PBSCalibrationPipeline::configure sample_fold->set_periodicity(" << nbin << "," << dat_offset << "," << dat_dec << ")" << endl;
  sample_fold->set_periodicity (nbin, dat_offset, dat_dec);
  sample_fold->set_verbose (verbose);
}

#ifdef HAVE_CUDA

void spip::PBSCalibrationPipeline::set_device (int _device)
{
  int device_count = 0;
  cudaError_t err = cudaGetDeviceCount(&device_count);
  if (err != cudaSuccess)
    throw runtime_error ("spip::PBSCalibrationPipeline::set_device cudaGetDeviceCount failed");

  device = _device;
  if (device >= device_count)
    throw runtime_error ("Specified CUDA device not available");

  // TODO error checking
  err = cudaSetDevice (device);
  if (err != cudaSuccess)
    throw runtime_error ("spip::PBSCalibrationPipeline::set_device cudaSetDevice failed");

  err = cudaStreamCreate(&stream);
   if (err != cudaSuccess)
    throw runtime_error ("spip::PBSCalibrationPipeline::set_device cudaStreamCreate failed");
}
  
//! build the pipeline containers and transforms
void spip::PBSCalibrationPipeline::configure_cuda (spip::UnpackFloat * unpacker)
{
  if (verbose)
    cerr << "spip::PBSCalibrationPipeline::configure_cuda creating input" << endl;
  // input container, reads header 
  input = new spip::ContainerRingRead (in_db);
  input->register_buffers();

  // transfer host to device
  d_input = new spip::ContainerCUDADevice ();

  if (verbose)
    cerr << "spip::PBSCalibrationPipeline::configure_cuda allocating RAM to CUDA Transfer" << endl;
  ram_to_cuda = new spip::RAMtoCUDATransfer (stream);
  ram_to_cuda->set_input (input);
  ram_to_cuda->set_output (d_input); 
  ram_to_cuda->set_verbose (verbose);

  if (verbose)
    cerr << "spip::PBSCalibrationPipeline::configure_cuda unpacked container" << endl;
  // unpacked container
  unpacked = new spip::ContainerCUDADevice ();
  if (verbose)
    cerr << "spip::PBSCalibrationPipeline::configure_cuda allocating UnpackFloat" << endl;

  // unpack to float operation
  if (verbose)
    cerr << "spip::PBSCalibrationPipeline::configure_cuda allocating UnpackFloat" << endl;
  unpack_float = unpacker;
  unpack_float->set_input (d_input);
  unpack_float->set_output (unpacked);
  unpack_float->set_verbose (verbose);
  UnpackFloatCUDA * tmp = dynamic_cast<UnpackFloatCUDA *>(unpacker);
  tmp->set_stream (stream);

  // output of sample fold
  if (verbose)
    cerr << "spip::PBSCalibrationPipeline::configure_cuda allocating folded container" << endl;
  d_output = new spip::ContainerCUDADevice ();

  // fold samples at the requested period
  if (verbose)
    cerr << "spip::PBSCalibrationPipeline::configure_cuda allocating Sample Fold" << endl;
  sample_fold = new spip::SampleFoldCUDA(stream);
  sample_fold->set_input (unpacked);
  sample_fold->set_output (d_output);
  sample_fold->set_periodicity (nbin, dat_offset, dat_dec);
  sample_fold->set_verbose (verbose);

  if (verbose)
    cerr << "spip::PBSCalibrationPipeline::configure_cuda allocating output Ring Buffer" << endl;
  // coarse output channels
  output = new spip::ContainerBufferedRingWrite (out_db);
  output->register_buffers();

  // transfer device to host
  if (verbose)
    cerr << "spip::PBSCalibrationPipeline::configure_cuda allocating CUDA to RAM Transfer" << endl;
  cuda_to_ram = new spip::CUDAtoRAMTransfer (stream);
  cuda_to_ram->set_input (d_output);
  cuda_to_ram->set_output (output);
  cuda_to_ram->set_verbose (verbose);
}

#endif

//! process meta-data through the pipeline, performing all resource allocation
void spip::PBSCalibrationPipeline::open ()
{
  if (verbose)
    cerr << "spip::PBSCalibrationPipeline::open()" << endl;

  if (verbose)
    cerr << "spip::PBSCalibrationPipeline::open input->read_header()" << endl;
  // read from the input
  input->process_header();

#ifdef HAVE_CUDA
  if (device >= 0)
  {
    if (verbose)
      cerr << "spip::PBSCalibrationPipeline::open ram_to_cuda->configure()" << endl;
    ram_to_cuda->configure(spip::Ordering::Custom);
  }
#endif
  
  // configure the unpacker
  if (verbose)
    cerr << "spip::PBSCalibrationPipeline::open unpack_float->configure()" << endl;
  unpack_float->set_scale (1.0f / 100.0f);
  unpack_float->configure(spip::Ordering::SFPT);

  // configure the sample fold
  if (verbose)
    cerr << "spip::PBSCalibrationPipeline::open sample_fold->configure()" << endl;
  sample_fold->configure(spip::Ordering::TSPFB);

#ifdef HAVE_CUDA
  if (device >= 0)
  {
    if (verbose)
      cerr << "spip::PBSCalibrationPipeline::open cuda_to_ram->configure()" << endl;
    cuda_to_ram->configure(spip::Ordering::TSPFB);
  }
#endif

  // write the output header
  if (verbose)
    cerr << "spip::PBSCalibrationPipeline::open output->write_header()" << endl;
  output->process_header();
}

//! close the input and output data blocks
void spip::PBSCalibrationPipeline::close ()
{
  if (verbose)
    cerr << "spip::PBSCalibrationPipeline::close()" << endl;

  if (in_db->is_block_open())
  {
    if (verbose)
      cerr << "spip::PBSCalibrationPipeline::close in_db->close_block(" << in_db->get_data_bufsz() << ")" << endl;
    in_db->close_block (in_db->get_data_bufsz());
  }

  // close the data blocks, ending the observation
  if (verbose)
    cerr << "spip::PBSCalibrationPipeline::close in_db->close()" << endl;
  in_db->close();

  if (verbose)
    cerr << "spip::PBSCalibrationPipeline::close out_db->close()" << endl;
  out_db->close();
}

// process blocks of input data until the end of the data stream
bool spip::PBSCalibrationPipeline::process ()
{
  if (verbose)
    cerr << "spip::PBSCalibrationPipeline::process ()" << endl;

  bool keep_processing = true;

  // commence observation on output data block
  if (verbose)
    cerr << "spip::PBSCalibrationPipeline::out_db->open()" << endl;
  out_db->open();

  uint64_t input_bufsz = in_db->get_data_bufsz();
  uint64_t nbytes_input;

  while (keep_processing)
  {
    // read a block of input data
    if (verbose)
      cerr << "spip::PBSCalibrationPipeline::process input->open_block()" << endl;
    nbytes_input = input->open_block();

    if (nbytes_input < input_bufsz)
      keep_processing = false;

    // only process full blocks of data
    if (keep_processing)
    {

  #ifdef HAVE_CUDA
      if (device >= 0)
      {
        ram_to_cuda->prepare();
        ram_to_cuda->transformation();
      }
  #endif

      if (verbose)
        cerr << "spip::PBSCalibrationPipeline::process unpack_float->transformation()" << endl;
      unpack_float->prepare();
      unpack_float->transformation();

      // perform Sample Fold
      if (verbose)
        cerr << "spip::PBSCalibrationPipeline::process sample_fold->transformation()" << endl;
      sample_fold->prepare();
      sample_fold->transformation ();

#ifdef HAVE_CUDA
      if (device >= 0)
      {
        if (verbose)
          cerr << "spip::PBSCalibrationPipeline::process cuda_to_ram->transformation()" << endl;
        cuda_to_ram->prepare();
        cuda_to_ram->transformation();
      }
#endif

      if (verbose)
        cerr << "spip::PBSCalibrationPipeline::process output->write_buffer()" << endl;
      output->write_buffer();
    }

    if (verbose)
      cerr << "spip::PBSCalibrationPipeline::process input->close_block()" << endl;
    input->close_block();
  }

  // close the data blocks
  close();

  return true;
}
