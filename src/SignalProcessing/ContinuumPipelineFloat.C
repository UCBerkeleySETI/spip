/***************************************************************************
 *
 *   Copyright (C) 2018 Andrew Jameson
 *   Licensed under the Academic Free License version 2.1
 *
 ***************************************************************************/

#include "spip/ContinuumPipelineFloat.h"

#include <signal.h>
#include <unistd.h>
#include <cstring>
#include <iostream>
#include <stdexcept>
#include <new>

//#define _DEBUG

using namespace std;

spip::ContinuumPipelineFloat::ContinuumPipelineFloat (const char * in_key_string, const char * out_dir_string)
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
#endif

  in_db  = new DataBlockRead (in_key_string);
  out_dir = string(out_dir_string);

  in_db->connect();
  in_db->lock();

  verbose = false;
}

spip::ContinuumPipelineFloat::~ContinuumPipelineFloat()
{
  in_db->unlock();
  in_db->disconnect();
  delete in_db;
}

void spip::ContinuumPipelineFloat::set_channelisation (int _nchan)
{
  nchan_out = _nchan;
  nfft = nchan_out * channel_oversampling;
  if (verbose) 
    cerr << "spip::ContinuumPipelineFloat::set_channelisation nchan_out=" << nchan_out << " nfft=" << nfft << endl;
}

void spip::ContinuumPipelineFloat::set_channel_oversampling (int _factor)
{
  if (verbose)
    cerr << "spip::ContinuumPipelineFloat::set_channel_oversampling factor=" << _factor << endl;
  channel_oversampling = _factor;
  nfft = nchan_out * channel_oversampling;
}

void spip::ContinuumPipelineFloat::set_decimation (int _tdec)
{
  if (verbose)
    cerr << "spip::ContinuumPipelineFloat::set_decimation tdec=" << _tdec << " fdec=" << channel_oversampling << endl;
  tdec = _tdec;
  fdec = channel_oversampling;
}

void spip::ContinuumPipelineFloat::set_tsubint (float _tsubint)
{
  if (verbose)
    cerr << "spip::ContinuumPipelineFloat::set_tsubint tsubint=" << _tsubint << endl;
  tsubint = _tsubint;
}

void spip::ContinuumPipelineFloat::set_output_state (spip::Signal::State _state)
{
  output_state = _state;
}

//! build the pipeline containers and transforms
void spip::ContinuumPipelineFloat::configure ()
{
  if (verbose)
    cerr << "spip::ContinuumPipelineFloat::configure ()" << endl;
#ifdef HAVE_CUDA
  if (device >= 0)
    return configure_cuda ();

  if (in_db->get_device() >= 0)
    throw runtime_error ("CPU pipeline cannot run on GPU datablock ");
#endif
  
  if (verbose)
    cerr << "spip::ContinuumPipelineFloat::configure creating input" << endl;
  // input containers, reads header 
  input = new spip::ContainerRingRead (in_db);

  if (verbose)
    cerr << "spip::ContinuumPipelineFloat::configure allocating reblocked container" << endl;
  reblocked = new spip::ContainerRAM ();

  if (verbose)
    cerr << "spip::ContinuumPipelineFloat::configure allocating RAMtoRAMTransfer" << endl;
  ram_to_ram = new spip::RAMtoRAMTransfer();
  ram_to_ram->set_input (input);
  ram_to_ram->set_output (reblocked);
  ram_to_ram->set_verbose (verbose);

  // fine channels
  if (verbose)
    cerr << "spip::ContinuumPipelineFloat::configure allocating channelised container" << endl;
  channelised = new spip::ContainerRAM ();

  if (verbose)
    cerr << "spip::ContinuumPipelineFloat::configure allocating Forward FFT" << endl;
  // forward FFT operation
  fwd_fft = new spip::ForwardFFTFFTW();
  fwd_fft->set_input (reblocked);
  fwd_fft->set_output (channelised);
  fwd_fft->set_nfft (nfft);
  fwd_fft->set_verbose (verbose);

  // detected data
  detected = new spip::ContainerRAM ();

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


  // lower sideband data
  if (verbose)
    cerr << "spip::ContinuumPipelineFloat::configure allocating output File Writer" << endl;
  output = new spip::ContainerRAMFileWrite (out_dir);

  // Temporal and spectral integrator 
  if (verbose)
    cerr << "spip::ContinuumPipelineFloat::configure integrator->set_decimation(" << tdec << ", 1,  " << fdec << ", 1)" << endl;
  integrator = new spip::IntegrationRAM();
  integrator->set_decimation (tdec, 1, fdec, 1);
  integrator->set_input (detected);
  integrator->set_output (output);
  integrator->set_verbose (verbose);
}

#ifdef HAVE_CUDA

void spip::ContinuumPipelineFloat::set_device (int _device)
{
  int device_count = 0;
  cudaError_t err = cudaGetDeviceCount(&device_count);
  if (err != cudaSuccess)
    throw runtime_error ("spip::ContinuumPipelineFloat::set_device cudaGetDeviceCount failed");

  device = _device;
  if (device >= device_count)
    throw runtime_error ("Specified CUDA device not available");

  if (in_db->get_device() != device)
    throw runtime_error ("Specified CUDA device did not match DB");

  err = cudaSetDevice (device);
  if (err != cudaSuccess)
    throw runtime_error ("spip::ContinuumPipelineFloat::set_device cudaSetDevice failed");

  err = cudaStreamCreate(&stream);
   if (err != cudaSuccess)
    throw runtime_error ("spip::ContinuumPipelineFloat::set_device cudaStreamCreate failed");
}
  
//! build the pipeline containers and transforms
void spip::ContinuumPipelineFloat::configure_cuda ()
{
  if (verbose)
    cerr << "spip::ContinuumPipelineFloat::configure_cuda creating input" << endl;
  // input container, reads header 
  input = new spip::ContainerRingRead (in_db);

  reblocked = new spip::ContainerCUDADevice ();

  if (verbose)
    cerr << "spip::ContinuumPipelineFloat::configure_cuda allocating RING to CUDA Transfer" << endl;
  ring_to_cuda = new spip::RINGtoCUDATransfer (stream);
  ring_to_cuda->set_input (input);
  ring_to_cuda->set_output (dynamic_cast<spip::ContainerCUDADevice*>(reblocked)); 
  ring_to_cuda->set_verbose (verbose);

  // fine channels
  if (verbose)
    cerr << "spip::ContinuumPipelineFloat::configure_cuda allocating channelised container" << endl;
  channelised = new spip::ContainerCUDADevice ();

  if (verbose)
    cerr << "spip::ContinuumPipelineFloat::configure_cuda allocating Forward FFT" << endl;
  // forward FFT operation
  fwd_fft = new spip::ForwardFFTCUDA(stream);
  fwd_fft->set_input (reblocked);
  fwd_fft->set_output (channelised);
  fwd_fft->set_nfft (nfft);
  fwd_fft->set_verbose (verbose);

  // detected data
  detected = new spip::ContainerCUDADevice();

  // coherennce detector
  if ((output_state == spip::Signal::Intensity) || (output_state == spip::Signal::PPQQ))
    detector = new spip::DetectionSquareLawCUDA(stream);
  else
    detector = new spip::DetectionPolarimetryCUDA(stream);
  detector->set_output_state (output_state);
  detector->set_input (channelised);
  detector->set_output (detected);
  detector->set_verbose (verbose);

  if (verbose)
    cerr << "spip::ContinuumPipelineFloat::configure_cuda allocating output File Writer" << endl;
  // coarse output channels
  d_output = new spip::ContainerCUDAFileWrite(stream, out_dir);

  // time and frequency integrator
  integrator = new spip::IntegrationCUDA(stream);
  integrator->set_decimation (tdec, 1, fdec, 1);
  integrator->set_input (detected);
  integrator->set_output (d_output);
  integrator->set_verbose (verbose);
}

#endif

//! process meta-data through the pipeline, performing all resource allocation
void spip::ContinuumPipelineFloat::open ()
{
  if (verbose)
    cerr << "spip::ContinuumPipelineFloat::open()" << endl;

  if (verbose)
    cerr << "spip::ContinuumPipelineFloat::open input->process_header()" << endl;
  // read from the input
  input->process_header();

  // now determine the required reblocking
  uint64_t input_block_size = input->get_size();
  uint64_t bytes_required = uint64_t(input->calculate_nbits_per_sample()) * nfft / 8;
  if (verbose)
    cerr << "spip::ContinuumPipelineFloat::open input->calculate_nbits_per_sample()=" 
         << input->calculate_nbits_per_sample() << " nfft=" << nfft << endl;
  if (bytes_required > input_block_size)
  {
    reblock_factor = bytes_required / input_block_size;
    if (bytes_required % input_block_size)
    {
      throw Error (InvalidState, "spip::ContinuumPipelineFloat::open", 
                   "bytes_required [%lu] % input_block_size [%lu] != 0\n", 
                    bytes_required, input_block_size);
    }
  }
  else
    reblock_factor = 1;

  if (verbose)
    cerr << "spip::ContinuumPipelineFloat::open input_block_size=" << input_block_size 
         << " bytes_required=" << bytes_required << " reblock_factor=" << reblock_factor << endl;
   
#ifdef HAVE_CUDA
  if (device >= 0)
  {
    if (verbose)
      cerr << "spip::ContinuumPipelineFloat::open ring_to_cuda->configure(SFPT)" << endl;
    ring_to_cuda->set_output_reblock (reblock_factor);
    ring_to_cuda->configure(spip::Ordering::SFPT);
  }
  else
#endif
  {
    if (verbose)
      cerr << "spip::ContinuumPipelineFloat::open ram_to_ram->configure(SFPT)" << endl;
    ram_to_ram->set_output_reblock (reblock_factor);
    ram_to_ram->configure (spip::Ordering::SFPT);
  }
  
  // configure the forward FFT
  if (verbose)
    cerr << "spip::ContinuumPipelineFloat::open fwd_fft->configure(TSPF)" << endl;
  fwd_fft->configure(spip::Ordering::TSPF);

  // configure detection
  if (verbose)
    cerr << "spip::ContinuumPipelineFloat::open detector->configure(TSPF)" << endl;
  detector->configure(spip::Ordering::TSPF);

  // configure integration
  if (verbose)
    cerr << "spip::ContinuumPipelineFloat::open integrator->configure(TSPF)" << endl;
  integrator->configure(spip::Ordering::TSPF);

  // write the output header
  if (verbose)
    cerr << "spip::ContinuumPipelineFloat::open output->process_header()" << endl;
#ifdef HAVE_CUDA
  if (device >= 0)
  {
    d_output->set_file_length_seconds (tsubint);
    d_output->process_header();
  }
  else
#endif
  {
    output->set_file_length_seconds (tsubint);
    output->process_header();
  }

}

//! close the input and output data blocks
void spip::ContinuumPipelineFloat::close ()
{
  if (verbose)
    cerr << "spip::ContinuumPipelineFloat::close()" << endl;

  if (in_db->is_block_open())
  {
    if (verbose)
      cerr << "spip::ContinuumPipelineFloat::close in_db->close_block(" << in_db->get_data_bufsz() << ")" << endl;
    in_db->close_block (in_db->get_data_bufsz());
  }

  // close the input data block
  if (verbose)
    cerr << "spip::ContinuumPipelineFloat::close in_db->close()" << endl;
  in_db->close();
}

// process blocks of input data until the end of the data stream
bool spip::ContinuumPipelineFloat::process ()
{
  if (verbose)
    cerr << "spip::ContinuumPipelineFloat::process ()" << endl;

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
        cerr << "spip::ContinuumPipelineFloat::process input->open_block()" << endl;
      nbytes_input = input->open_block();

      if (nbytes_input < input_bufsz)
        keep_processing = false;

      // only process full blocks of data
      if (keep_processing)
      {
#ifdef HAVE_CUDA
        if (device >= 0)
        {
          ring_to_cuda->prepare();
          ring_to_cuda->transformation();
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
        cerr << "spip::ContinuumPipelineFloat::process input->close_block()" << endl;
      input->close_block();
    }

    if (keep_processing)
    {
      // perform Forward FFT operation
      if (verbose)
        cerr << "spip::ContinuumPipelineFloat::process fwd_fft->transformation()" << endl;
      fwd_fft->prepare();
      fwd_fft->transformation ();

      if (verbose)
        cerr << "spip::ContinuumPipelineFloat::process detector->transformation()" << endl;
      detector->prepare();
      detector->transformation();

      if (verbose)
        cerr << "spip::ContinuumPipelineFloat::process integrator->transformation()" << endl;
      integrator->prepare();
      integrator->transformation();

#ifdef HAVE_CUDA
      if (device >= 0)
      {
        uint64_t output_ndat = d_output->get_ndat();
        if (output_ndat > 0)
        {
          if (verbose)
            cerr << "spip::ContinuumPipelineFloat::process writing " 
                 << output_ndat << " samples" << endl;
          d_output->write (output_ndat);        
        }
        else
        {
          if (verbose)
            cerr << "spip::ContinuumPipelineFloat::process no samples to write" << endl;
        }
      }
      else
#endif
      {
        uint64_t output_ndat = output->get_ndat();
        if (output_ndat > 0)
        {
          cerr << "spip::ContinuumPipelineFloat::process writing " 
               << output_ndat << " samples" << endl;
          output->write (output_ndat);        
        }
        else
        {
          if (verbose)
            cerr << "spip::ContinuumPipelineFloat::process no samples to write" << endl;
        }
      }
    }
  }

  // close the data blocks
  close();

  return true;
} 
