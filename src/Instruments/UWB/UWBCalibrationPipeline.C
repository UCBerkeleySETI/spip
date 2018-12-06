/***************************************************************************
 *
 *   Copyright (C) 2018 Andrew Jameson
 *   Licensed under the Academic Free License version 2.1
 *
 ***************************************************************************/

#include "spip/UWBCalibrationPipeline.h"

#include <signal.h>
#include <unistd.h>
#include <cstring>
#include <iostream>
#include <stdexcept>
#include <new>

//#define _DEBUG

using namespace std;

spip::UWBCalibrationPipeline::UWBCalibrationPipeline (const char * in_key_string, const char * out_key_string)
{
  output_state = spip::Signal::Coherence;

  nfft = 128;
  dat_dec = 1;
  pol_dec = 1;
  chan_dec = 1;
  signal_dec = 1;

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

spip::UWBCalibrationPipeline::~UWBCalibrationPipeline()
{
  in_db->unlock();
  in_db->disconnect();
  delete in_db;

  out_db->unlock();
  out_db->disconnect();
  delete out_db;
}

void spip::UWBCalibrationPipeline::set_channelisation (unsigned _nfft)
{
  if (verbose)
    cerr << "spip::UWBCalibrationPipeline::set_channelisation nfft=" << _nfft << endl;
  nfft = _nfft;
}

void spip::UWBCalibrationPipeline::set_decimation (uint64_t _dat_dec, unsigned _pol_dec, unsigned _chan_dec)
{
  if (verbose) 
    cerr << "spip::UWBCalibrationPipeline::set_periodicity dat_dec=" << _dat_dec << " pol_dec=" << _pol_dec << endl;
  dat_dec = _dat_dec;
  pol_dec = _pol_dec;
  chan_dec = _chan_dec;
  signal_dec = 1;
}

void spip::UWBCalibrationPipeline::set_output_state (spip::Signal::State _state)
{
  output_state = _state;
}

//! build the pipeline containers and transforms
void spip::UWBCalibrationPipeline::configure ()
{
  if (verbose)
    cerr << "spip::UWBCalibrationPipeline::configure ()" << endl;
#ifdef HAVE_CUDA
  if (device >= 0)
    return configure_cuda();
#endif
  
  if (verbose)
    cerr << "spip::UWBCalibrationPipeline::configure creating input" << endl;
  // input containers, reads header 
  input = new spip::ContainerRingRead (in_db);

  // finer channels
  if (verbose)
    cerr << "spip::ContinuumPipeline::configure allocating channelised container" << endl;
  channelised = new spip::ContainerRAM ();

  if (verbose)
    cerr << "spip::ContinuumPipeline::configure allocating Forward FFT" << endl;
  // forward FFT operation
  fwd_fft = new spip::ForwardFFTFFTW();
  fwd_fft->set_input (input);
  fwd_fft->set_output (channelised);
  fwd_fft->set_nfft (nfft);
  fwd_fft->set_verbose (verbose);


  // detected data
  if (verbose)
    cerr << "spip::ContinuumPipeline::configure allocating detected container" << endl;
  detected = new spip::ContainerRAM ();

  if (verbose)
    cerr << "spip::ContinuumPipeline::configure allocating DetectionSquareLaw" << endl;
  // detection operation
  detection = new spip::DetectionSquareLawRAM();
  detection->set_input (channelised);
  detection->set_output (detected);
  detection->set_output_state (spip::Signal::PPQQ);
  detection->set_verbose (verbose);

  if (verbose)
    cerr << "spip::UWBCalibrationPipeline::configure allocating output Ring Buffer" << endl;
  output = new spip::ContainerBufferedRingWrite (out_db);

  if (verbose)
    cerr << "spip::UWBCalibrationPipeline::configure allocating IntergrationBinned" << endl;
  // sample fold operation
  integration_binned = new spip::IntegrationBinnedRAM();
  integration_binned->set_input (detected);
  integration_binned->set_output (output);
  integration_binned->set_decimation (dat_dec, pol_dec, chan_dec, signal_dec);
  integration_binned->set_verbose (verbose);
}

#ifdef HAVE_CUDA

void spip::UWBCalibrationPipeline::set_device (int _device)
{
  int device_count = 0;
  cudaError_t err = cudaGetDeviceCount(&device_count);
  if (err != cudaSuccess)
    throw runtime_error ("spip::UWBCalibrationPipeline::set_device cudaGetDeviceCount failed");

  device = _device;
  if (device >= device_count)
    throw runtime_error ("Specified CUDA device not available");

  // TODO error checking
  err = cudaSetDevice (device);
  if (err != cudaSuccess)
    throw runtime_error ("spip::UWBCalibrationPipeline::set_device cudaSetDevice failed");

  err = cudaStreamCreate(&stream);
   if (err != cudaSuccess)
    throw runtime_error ("spip::UWBCalibrationPipeline::set_device cudaStreamCreate failed");
}
  
//! build the pipeline containers and transforms
void spip::UWBCalibrationPipeline::configure_cuda ()
{
  if (verbose)
    cerr << "spip::UWBCalibrationPipeline::configure_cuda creating input" << endl;
  // input container, reads header 
  d_input = new spip::ContainerRingReadCUDA (in_db);

  // output of forward fft
  if (verbose)
    cerr << "spip::UWBCalibrationPipeline::configure_cuda allocating channelised container" << endl;
  channelised  = new spip::ContainerCUDADevice ();

  if (verbose)
    cerr << "spip::ContinuumPipeline::configure_cuda allocating Forward FFT" << endl;
  // forward FFT operation
  fwd_fft = new spip::ForwardFFTCUDA(stream);
  fwd_fft->set_input (d_input);
  fwd_fft->set_output (channelised);
  fwd_fft->set_nfft (nfft);
  fwd_fft->set_verbose (verbose);

  // detected data
  if (verbose)
    cerr << "spip::ContinuumPipeline::configure_cuda allocating detected container" << endl;
  detected = new spip::ContainerCUDADevice();

  if (verbose)
    cerr << "spip::ContinuumPipeline::configure_cuda allocating DetectionSquareLaw" << endl;
  // forward FFT operation
  detection = new spip::DetectionSquareLawCUDA(stream);
  detection->set_input (channelised);
  detection->set_output (detected);
  detection->set_output_state (spip::Signal::PPQQ);
  detection->set_verbose (verbose);

  // output of integration binned
  d_output = new spip::ContainerCUDADevice ();

  if (verbose)
    cerr << "spip::UWBCalibrationPipeline::configure_cuda allocating IntegrationBinnedCUDA" << endl;
  integration_binned = new spip::IntegrationBinnedCUDA(stream);
  integration_binned->set_input (detected);
  integration_binned->set_output (d_output);
  integration_binned->set_decimation (dat_dec, pol_dec, chan_dec, signal_dec);
  integration_binned->set_verbose (verbose);

  if (verbose)
    cerr << "spip::UWBCalibrationPipeline::configure_cuda allocating output Ring Buffer" << endl;
  // coarse output channels
  output = new spip::ContainerBufferedRingWrite (out_db);
  output->register_buffers();

  // transfer device to host
  if (verbose)
    cerr << "spip::UWBCalibrationPipeline::configure_cuda allocating CUDA to RAM Transfer" << endl;
  cuda_to_ram = new spip::CUDAtoRAMTransfer (stream);
  cuda_to_ram->set_input (d_output);
  cuda_to_ram->set_output (output);
  cuda_to_ram->set_verbose (verbose);
}

#endif

//! process meta-data through the pipeline, performing all resource allocation
void spip::UWBCalibrationPipeline::open ()
{
  if (verbose)
    cerr << "spip::UWBCalibrationPipeline::open()" << endl;

  if (verbose)
    cerr << "spip::UWBCalibrationPipeline::open input->read_header()" << endl;
  // read from the input

#ifdef HAVE_CUDA
  if (device >= 0)
  {
    d_input->process_header();
  }
  else
#endif
  { 
    input->process_header();
  }

  // configure the forward fft
  if (verbose)
    cerr << "spip::UWBCalibrationPipeline::open fwd_fft->configure()" << endl;
  fwd_fft->configure(spip::Ordering::TSPF);

  // configure the detection
  if (verbose)
    cerr << "spip::UWBCalibrationPipeline::open detection->configure()" << endl;
  detection->configure (spip::Ordering::TSPF);

  // configure the sample fold
  if (verbose)
    cerr << "spip::UWBCalibrationPipeline::open integration_binned->configure()" << endl;
  integration_binned->configure(spip::Ordering::TSPFB);

#ifdef HAVE_CUDA
  if (device >= 0)
  {
    if (verbose)
      cerr << "spip::UWBCalibrationPipeline::open cuda_to_ram->configure()" << endl;
    cuda_to_ram->configure(spip::Ordering::TSPFB);
  }
#endif

  // write the output header
  if (verbose)
    cerr << "spip::UWBCalibrationPipeline::open output->write_header()" << endl;
  output->process_header();
}

//! close the input and output data blocks
void spip::UWBCalibrationPipeline::close ()
{
  if (verbose)
    cerr << "spip::UWBCalibrationPipeline::close()" << endl;

  if (in_db->is_block_open())
  {
    if (verbose)
      cerr << "spip::UWBCalibrationPipeline::close in_db->close_block(" << in_db->get_data_bufsz() << ")" << endl;
    in_db->close_block (in_db->get_data_bufsz());
  }

  // close the data blocks, ending the observation
  if (verbose)
    cerr << "spip::UWBCalibrationPipeline::close in_db->close()" << endl;
  in_db->close();

  if (verbose)
    cerr << "spip::UWBCalibrationPipeline::close out_db->close()" << endl;
  out_db->close();
}

// process blocks of input data until the end of the data stream
bool spip::UWBCalibrationPipeline::process ()
{
  if (verbose)
    cerr << "spip::UWBCalibrationPipeline::process ()" << endl;

  bool keep_processing = true;

  // commence observation on output data block
  if (verbose)
    cerr << "spip::UWBCalibrationPipeline::out_db->open()" << endl;
  out_db->open();

  uint64_t input_bufsz = in_db->get_data_bufsz();
  uint64_t nbytes_input;

  while (keep_processing)
  {
    // read a block of input data
    if (verbose)
      cerr << "spip::UWBCalibrationPipeline::process input->open_block()" << endl;
#ifdef HAVE_CUDA
    if (device >= 0)
    {
      nbytes_input = d_input->open_block();
    }
    else
#endif
    {
      nbytes_input = input->open_block();
    }
    if (verbose)
      cerr << "spip::UWBCalibrationPipeline::process nbytes_input=" << nbytes_input << endl;

    if (nbytes_input < input_bufsz)
      keep_processing = false;

    // only process full blocks of data
    if (keep_processing)
    {
      if (verbose)
        cerr << "spip::UWBCalibrationPipeline::process fwd_fft->transformation()" << endl;
      fwd_fft->prepare();
      fwd_fft->transformation();

      if (verbose)
        cerr << "spip::UWBCalibrationPipeline::process detection->transformation()" << endl;
      detection->prepare();
      detection->transformation();

      // perform Sample Fold
      if (verbose)
        cerr << "spip::UWBCalibrationPipeline::process integration_binned->transformation()" << endl;
      integration_binned->prepare();
      integration_binned->transformation ();

#ifdef HAVE_CUDA
      if (device >= 0)
      {
        if (verbose)
          cerr << "spip::UWBCalibrationPipeline::process cuda_to_ram->transformation()" << endl;
        cuda_to_ram->prepare();
        cuda_to_ram->transformation();
      }
#endif

      if (verbose)
        cerr << "spip::UWBCalibrationPipeline::process output->write_buffer()" << endl;
      output->write_buffer();
    }

    if (verbose)
      cerr << "spip::UWBCalibrationPipeline::process input->close_block()" << endl;
#ifdef HAVE_CUDA
    if (device >= 0)
    {
      input->close_block();
    }
    else
#endif
    {
      input->close_block();
    }
  }

  // close the data blocks
  close();

  return true;
}
