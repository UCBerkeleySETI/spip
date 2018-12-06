/***************************************************************************
 *
 *   Copyright (C) 2016 Andrew Jameson
 *   Licensed under the Academic Free License version 2.1
 *
 ***************************************************************************/

#include "spip/AdaptiveFilterTest.h"

#include <signal.h>
#include <unistd.h>
#include <cstring>
#include <iostream>
#include <stdexcept>
#include <new>

//#define _DEBUG

using namespace std;

spip::AdaptiveFilterTest::AdaptiveFilterTest (const char * in_key_string, const char * out_key_string)
{
#ifdef HAVE_CUDA
  device = -1;
#endif

  in_db  = new DataBlockRead (in_key_string);
  out_db = new DataBlockWrite (out_key_string);

  in_db->connect();
  in_db->lock();

  out_db->connect();
  out_db->lock();

  reference_pol = -1;
  verbose = false;
  req_mon_tsamp = 0;
}

spip::AdaptiveFilterTest::~AdaptiveFilterTest()
{
  in_db->unlock();
  in_db->disconnect();
  delete in_db;

  out_db->unlock();
  out_db->disconnect();
  delete out_db;
}

void spip::AdaptiveFilterTest::set_filtering (int ref_pol, double _req_mon_tsamp)
{
  reference_pol = ref_pol;
  req_mon_tsamp = _req_mon_tsamp;
}

//! build the pipeline containers and transforms
void spip::AdaptiveFilterTest::configure (spip::UnpackFloat * unpacker)
{
  if (verbose)
    cerr << "spip::AdaptiveFilterTest::configure ()" << endl;
#ifdef HAVE_CUDA
  if (device >= 0)
    return configure_cuda(unpacker);
#endif
  
  if (verbose)
    cerr << "spip::AdaptiveFilterTest::configure creating input" << endl;
  // input containers, reads header 
  input = new spip::ContainerRingRead (in_db);

  if (verbose)
    cerr << "spip::AdaptiveFilterTest::configure unpacked container" << endl;
  // unpacked containers
  unpacked = new spip::ContainerRAM ();
  
  if (verbose)
    cerr << "spip::AdaptiveFilterTest::configure allocating UnpackFloat" << endl;
  // unpack to float operation
  unpack_float = unpacker;
  unpack_float->set_input (input);
  unpack_float->set_output (unpacked);
  unpack_float->set_verbose (verbose);

  // output ring buffer
  if (verbose)
    cerr << "spip::AdaptiveFilterTest::configure allocating output Ring Buffer" << endl;
  output = new spip::ContainerRingWrite (out_db);
  
  // TODO parameterise this
  string output_dir = string(".");

  // RFI Filtering operation
  filter = new spip::AdaptiveFilterRAM(output_dir);
  filter->set_input (unpacked);
  filter->set_output (output);
  filter->set_filtering (reference_pol, req_mon_tsamp);
  filter->set_verbose (verbose);
}

#ifdef HAVE_CUDA

void spip::AdaptiveFilterTest::set_device (int _device)
{
  int device_count = 0;
  cudaError_t err = cudaGetDeviceCount(&device_count);
  if (err != cudaSuccess)
    throw runtime_error ("spip::AdaptiveFilterTest::set_device cudaGetDeviceCount failed");

  device = _device;
  if (device >= device_count)
    throw runtime_error ("Specified CUDA device not available");

  // TODO error checking
  err = cudaSetDevice (device);
  if (err != cudaSuccess)
    throw runtime_error ("spip::AdaptiveFilterTest::set_device cudaSetDevice failed");

  err = cudaStreamCreate(&stream);
   if (err != cudaSuccess)
    throw runtime_error ("spip::AdaptiveFilterTest::set_device cudaStreamCreate failed");
}
  
//! build the pipeline containers and transforms
void spip::AdaptiveFilterTest::configure_cuda (spip::UnpackFloat * unpacker)
{
  if (verbose)
    cerr << "spip::AdaptiveFilterTest::configure_cuda creating input" << endl;
  // input container, reads header 
  input = new spip::ContainerRingRead (in_db);
  input->register_buffers();

  // transfer host to device
  d_input = new spip::ContainerCUDADevice ();

  if (verbose)
    cerr << "spip::AdaptiveFilterTest::configure_cuda allocating RAM to CUDA Transfer" << endl;
  ram_to_cuda = new spip::RAMtoCUDATransfer (stream);
  ram_to_cuda->set_input (input);
  ram_to_cuda->set_output (d_input); 
  ram_to_cuda->set_verbose (verbose);

  if (verbose)
    cerr << "spip::AdaptiveFilterTest::configure_cuda unpacked container" << endl;
  // unpacked container
  unpacked = new spip::ContainerCUDADevice ();
  if (verbose)
    cerr << "spip::AdaptiveFilterTest::configure_cuda allocating UnpackFloat" << endl;

  // unpack to float operation
  unpack_float = unpacker;
  unpack_float->set_input (d_input);
  unpack_float->set_output (unpacked);
  unpack_float->set_verbose (verbose);
  UnpackFloatCUDA * tmp = dynamic_cast<UnpackFloatCUDA *>(unpacker);
  if (tmp)
    tmp->set_stream (stream);
  else
    throw Error (InvalidState, "spip::AdaptiveFilterTest::configure_cuda", "unpacker must be a UnpackFloatCUDA");

  // cleaned data
  d_output = new spip::ContainerCUDADevice ();

  // TODO parameterise this
  string output_dir = string(".");

  // RFI Filtering operation
  filter = new spip::AdaptiveFilterCUDA(stream, output_dir);
  filter->set_input (unpacked);
  filter->set_output (d_output);
  filter->set_filtering (reference_pol, req_mon_tsamp);
  filter->set_verbose (verbose);

  if (verbose)
    cerr << "spip::AdaptiveFilterTest::configure_cuda allocating output Ring Buffer" << endl;
  // coarse output channels
  output = new spip::ContainerRingWrite (out_db);
  output->register_buffers();

  // transfer device to host
  if (verbose)
    cerr << "spip::AdaptiveFilterTest::configure_cuda allocating CUDA to RAM Transfer" << endl;
  cuda_to_ram = new spip::CUDAtoRAMTransfer (stream);
  cuda_to_ram->set_input (d_output);
  cuda_to_ram->set_output (output);
  cuda_to_ram->set_verbose (verbose);
}

#endif

//! process meta-data through the pipeline, performing all resource allocation
void spip::AdaptiveFilterTest::open ()
{
  if (verbose)
    cerr << "spip::AdaptiveFilterTest::open()" << endl;

  if (verbose)
    cerr << "spip::AdaptiveFilterTest::open input->read_header()" << endl;
  // read from the input
  input->process_header();

#ifdef HAVE_CUDA
  if (device >= 0)
  {
    if (verbose)
      cerr << "spip::AdaptiveFilterTest::open ram_to_cuda->configure()" << endl;
    ram_to_cuda->configure(spip::Ordering::SFPT);
  }
#endif
  
  // configure the unpacker
  if (verbose)
    cerr << "spip::AdaptiveFilterTest::open unpack_float->configure()" << endl;
  unpack_float->configure(spip::Ordering::SFPT);

  if (verbose)
    cerr << "spip::AdaptiveFilterTest::open filter->configure()" << endl;
  filter->configure(spip::Ordering::SFPT);

#ifdef HAVE_CUDA
  if (device >= 0)
  {
    if (verbose)
      cerr << "spip::AdaptiveFilterTest::open cuda_to_ram->configure()" << endl;
    cuda_to_ram->configure(spip::Ordering::SFPT);
  }
#endif

  // write the output header
  if (verbose)
    cerr << "spip::AdaptiveFilterTest::open output->write_header()" << endl;
  output->process_header();
}

//! close the input and output data blocks
void spip::AdaptiveFilterTest::close ()
{
  if (verbose)
    cerr << "spip::AdaptiveFilterTest::close()" << endl;

  if (out_db->is_block_open())
  {
    if (verbose)
      cerr << "spip::AdaptiveFilterTest::close out_db->close_block(" << out_db->get_data_bufsz() << ")" << endl;
    out_db->close_block (out_db->get_data_bufsz());
  }

  if (in_db->is_block_open())
  {
    if (verbose)
      cerr << "spip::AdaptiveFilterTest::close in_db->close_block(" << in_db->get_data_bufsz() << ")" << endl;
    in_db->close_block (in_db->get_data_bufsz());
  }

  // close the data blocks, ending the observation
  if (verbose)
    cerr << "spip::AdaptiveFilterTest::close in_db->close()" << endl;
  in_db->close();

  if (verbose)
    cerr << "spip::AdaptiveFilterTest::close out_db->close()" << endl;
  out_db->close();
}

// process blocks of input data until the end of the data stream
bool spip::AdaptiveFilterTest::process ()
{
  if (verbose)
    cerr << "spip::AdaptiveFilterTest::process ()" << endl;

  bool keep_processing = true;

  // commence observation on output data block
  if (verbose)
    cerr << "spip::AdaptiveFilterTest::out_db->open()" << endl;
  out_db->open();

  uint64_t input_bufsz = in_db->get_data_bufsz();
  uint64_t nbytes;

  unsigned blocks_processed = 0;
  unsigned blocks_per_mon_tsamp = filter->get_blocks_per_mon_tsamp();

  while (keep_processing)
  {
    // read a block of input data
    if (verbose)
      cerr << "spip::AdaptiveFilterTest::process input->open_block()" << endl;
    nbytes = input->open_block();

    if (nbytes < input_bufsz)
      keep_processing = false;

    // open a block of output data
    if (verbose)
      cerr << "spip::AdaptiveFilterTest::process output->open_block()" << endl;
    output->open_block();

#ifdef HAVE_CUDA
  if (device >= 0)
  {
    ram_to_cuda->prepare();
    ram_to_cuda->transformation();
  }
#endif

    if (verbose)
      cerr << "spip::AdaptiveFilterTest::process unpack_float->transformation()" << endl;
    unpack_float->prepare();
    unpack_float->transformation();

    if (verbose)
      cerr << "spip::AdaptiveFilterTest::process filter->transformation()" << endl;
    filter->prepare();
    filter->transformation();

    // count the bytes processed
    blocks_processed++;
    if (blocks_processed % blocks_per_mon_tsamp == 0)
    {
      filter->write_gains();
      filter->write_dirty();
      filter->write_cleaned();
    }

#ifdef HAVE_CUDA
    if (device >= 0)
    {
      cuda_to_ram->prepare();
      cuda_to_ram->transformation();
    }
#endif
    if (verbose)
      cerr << "spip::AdaptiveFilterTest::process input->close_block()" << endl;
    input->close_block();
    if (verbose)
      cerr << "spip::AdaptiveFilterTest::process output->close_block()" << endl;
    output->close_block();
  }

  // close the data blocks
  close();

  return true;
}
