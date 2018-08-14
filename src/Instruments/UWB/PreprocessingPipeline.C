/***************************************************************************
 *
 *   Copyright (C) 2018 Andrew Jameson
 *   Licensed under the Academic Free License version 2.1
 *
 ***************************************************************************/

#include "spip/PreprocessingPipeline.h"

#include <signal.h>
#include <unistd.h>
#include <cstring>
#include <iostream>
#include <stdexcept>
#include <new>

//#define _DEBUG

using namespace std;

spip::PreprocessingPipeline::PreprocessingPipeline (const char * in_key_string, 
                                                    const char * cal_key_string,
                                                    const char * trans_key_string,
                                                    const char * out_key_string)
{
  output_state = spip::Signal::Analytic;
  cal_output_state = spip::Signal::Intensity;

  nfft = 128;

  cal_dat_dec = 1;
  cal_pol_dec = 1;
  cal_chan_dec = 1;
  cal_signal_dec = 1;

  trans_dat_dec = 1;
  trans_pol_dec = 1;
  trans_chan_dec = 1;
  trans_signal_dec = 1;

  ref_pol = 0;

#ifdef HAVE_CUDA
  device = -1;
#endif
  calibrate = false;
  transients = false;
  filter = false;

  unpack_float = NULL;
  pol_sel = NULL;
  fwd_fft = NULL;
  adap_fil = NULL;
  bwd_fft = NULL;
  detect = NULL;
  integrate_cal = NULL;
  integrate_trans = NULL;
  input = NULL;
  unpacked = NULL;
  channelised = NULL;
  filtered = NULL;
  cal_output = NULL;
  trans_output = NULL;
  output = NULL;

#ifdef HAVE_CUDA
  d_output = NULL;
  ram_to_cuda = NULL;
  d_input = NULL;
  d_cal_output = NULL;
  d_trans_output = NULL;
  cuda_to_ram_cal = NULL;
  cuda_to_ram_trans = NULL;
#endif

  in_db  = new DataBlockRead (in_key_string);
  cal_db = new DataBlockWrite (cal_key_string);
  out_db = new DataBlockWrite (out_key_string);
  trans_db = new DataBlockWrite (trans_key_string);

  in_db->connect();
  in_db->lock();

  cal_db->connect();
  cal_db->lock();

  out_db->connect();
  out_db->lock();

  trans_db->connect();
  trans_db->lock();

  verbose = false;
}

spip::PreprocessingPipeline::~PreprocessingPipeline()
{
  in_db->unlock();
  in_db->disconnect();
  delete in_db;

  cal_db->unlock();
  cal_db->disconnect();
  delete cal_db;

  out_db->unlock();
  out_db->disconnect();
  delete out_db;

  trans_db->unlock();
  trans_db->disconnect();
  delete trans_db;
}

//! total number of polarisations and the reference pol (-1 for no ref pol)
void spip::PreprocessingPipeline::set_filtering (int _ref_pol)
{
  ref_pol = _ref_pol;
}

void spip::PreprocessingPipeline::set_channelisation (unsigned _nfft)
{
  if (verbose)
    cerr << "spip::PreprocessingPipeline::set_channelisation nfft=" << _nfft << endl;
  nfft = _nfft;
}

void spip::PreprocessingPipeline::set_cal_decimation (unsigned _chan_dec, uint64_t _dat_dec, unsigned _pol_dec)
{
  if (verbose) 
    cerr << "spip::PreprocessingPipeline::set_cal_decimation chan_dec=" << _chan_dec << " dat_dec=" << _dat_dec << " pol_dec=" << _pol_dec << endl;
  cal_dat_dec = _dat_dec;
  cal_pol_dec = _pol_dec;
  cal_chan_dec = _chan_dec;
  cal_signal_dec = 1;
}

void spip::PreprocessingPipeline::set_trans_decimation (uint64_t _dat_dec, unsigned _pol_dec)
{
  if (verbose)
    cerr << "spip::PreprocessingPipeline::set_trans_decimation dat_dec=" << _dat_dec << " pol_dec=" << _pol_dec << endl;
  trans_dat_dec = _dat_dec;
  trans_pol_dec = _pol_dec;
  trans_chan_dec = 1;
  trans_signal_dec = 1;
}

void spip::PreprocessingPipeline::set_function (bool _calibrate, bool _filter, bool _transients)
{
  calibrate = _calibrate;
  filter = _filter;
  transients = _transients;
}

//! build the pipeline containers and transforms
void spip::PreprocessingPipeline::configure (spip::UnpackFloat * unpacker)
{
  if (verbose)
    cerr << "spip::PreprocessingPipeline::configure ()" << endl;
#ifdef HAVE_CUDA
  if (device >= 0)
    return configure_cuda(unpacker);
#endif
  
  if (verbose)
    cerr << "spip::PreprocessingPipeline::configure creating input" << endl;
  // input containers, reads header 
  input = new spip::ContainerRingRead (in_db);

  if (verbose)
    cerr << "spip::PreprocessingPipeline::configure unpacked container" << endl;
  unpacked = new spip::ContainerRAM ();
  
  if (calibrate)
  {
    if (verbose)
      cerr << "spip::PreprocessingPipeline::configure allocating CAL output Ring Buffer" << endl;
    cal_output = new spip::ContainerBufferedRingWrite (cal_db);
  }

  if (transients)
  {
    if (verbose)
      cerr << "spip::PreprocessingPipeline::configure allocating TRANS output Ring Buffer" << endl;
    trans_output = new spip::ContainerRingWrite (trans_db);
  }

  if (verbose)
    cerr << "spip::PreprocessingPipeline::configure allocating output Ring Buffer" << endl;
  output = new spip::ContainerRingWrite (out_db);

  if (verbose)
    cerr << "spip::PreprocessingPipeline::configure allocating UnpackFloat" << endl;
  // unpack to float operation
  unpack_float = unpacker;
  unpack_float->set_input (input);
  unpack_float->set_output (unpacked);
  unpack_float->set_verbose (verbose);

  // if no filtering is required, remove excess polarisations
  if (!filter)
  {
    // Pol Select 
    pol_sel = new spip::PolSelectRAM();
    if (ref_pol > 0)
      pol_sel->set_pol_reduction ();
    pol_sel->set_input (unpacked);
    pol_sel->set_output (output);
    pol_sel->set_verbose (verbose);

    if (calibrate || transients)
    {
      channelised = new spip::ContainerRAM ();

      fwd_fft = new spip::ForwardFFTFFTW();
      fwd_fft->set_input (output);
      fwd_fft->set_output (channelised);
      fwd_fft->set_nfft (nfft);
      fwd_fft->set_verbose (verbose);

      detected = new spip::ContainerRAM ();

      detect = new spip::DetectionSquareLawRAM();
      detect->set_input (channelised);
      detect->set_output (detected);
      detect->set_output_state (spip::Signal::Intensity);
      detect->set_verbose (verbose);
    }

    if (calibrate)
    {
      integrate_cal = new spip::IntegrationBinnedRAM();
      integrate_cal->set_input (detected);
      integrate_cal->set_output (cal_output);
      integrate_cal->set_decimation (cal_dat_dec, cal_pol_dec, cal_chan_dec, cal_signal_dec);
      integrate_cal->set_verbose (verbose);
    }

    if (transients)
    {
      integrate_trans = new spip::IntegrationRAM();
      integrate_trans->set_input (detected);
      integrate_trans->set_output (trans_output); 
      integrate_trans->set_decimation (trans_dat_dec, trans_pol_dec, trans_chan_dec, trans_signal_dec);
      integrate_trans->set_verbose (verbose);
    }
  }
  else
  {
    channelised = new spip::ContainerRAM ();

    // forward FFT operation
    fwd_fft = new spip::ForwardFFTFFTW();
    fwd_fft->set_input (unpacked);
    fwd_fft->set_output (channelised);
    fwd_fft->set_nfft (nfft);
    fwd_fft->set_verbose (verbose);

    // cleaned data
    filtered = new spip::ContainerRAM ();

    // TODO parameterise
    string output_dir = string(".");

    // The filtering pipeline will filter pols 1+2 against pol3
    adap_fil = new spip::AdaptiveFilterRAM(output_dir);
    adap_fil->set_input (channelised);
    adap_fil->set_output (filtered);
    adap_fil->set_filtering (ref_pol);
    adap_fil->set_verbose (verbose);

    if (calibrate || transients)
    {
      detected = new spip::ContainerRAM ();

      detect = new spip::DetectionSquareLawRAM();
      detect->set_input (filtered);
      detect->set_output (detected);
      detect->set_output_state (spip::Signal::Intensity);
      detect->set_verbose (verbose);
    }

    if (calibrate)
    {
      integrate_cal = new spip::IntegrationBinnedRAM();
      integrate_cal->set_input (detected);
      integrate_cal->set_output (cal_output);
      integrate_cal->set_decimation (cal_dat_dec, cal_pol_dec, cal_chan_dec, cal_signal_dec);
      integrate_cal->set_verbose (verbose);
    }

    if (transients)
    {
      integrate_trans = new spip::IntegrationRAM();
      integrate_trans->set_input (detected);
      integrate_trans->set_output (trans_output);
      integrate_trans->set_decimation (trans_dat_dec, trans_pol_dec, trans_chan_dec, trans_signal_dec);
      integrate_trans->set_verbose (verbose);
    }

    bwd_fft = new spip::BackwardFFTFFTW();
    bwd_fft->set_input (filtered);
    bwd_fft->set_output (output);
    bwd_fft->set_nfft (nfft);
    bwd_fft->set_verbose (verbose);
  }
}

#ifdef HAVE_CUDA

void spip::PreprocessingPipeline::set_device (int _device)
{
  int device_count = 0;
  cudaError_t err = cudaGetDeviceCount(&device_count);
  if (err != cudaSuccess)
    throw runtime_error ("spip::PreprocessingPipeline::set_device cudaGetDeviceCount failed");

  device = _device;
  if (device >= device_count)
    throw runtime_error ("Specified CUDA device not available");

  // TODO error checking
  err = cudaSetDevice (device);
  if (err != cudaSuccess)
    throw runtime_error ("spip::PreprocessingPipeline::set_device cudaSetDevice failed");

  err = cudaStreamCreate(&stream);
   if (err != cudaSuccess)
    throw runtime_error ("spip::PreprocessingPipeline::set_device cudaStreamCreate failed");
}
  
//! build the pipeline containers and transforms
void spip::PreprocessingPipeline::configure_cuda (spip::UnpackFloat * unpacker)
{
  //spip::Container::verbose = true;

  if (verbose)
    cerr << "spip::PreprocessingPipeline::configure_cuda creating input" << endl;
  // input container, reads header 
  input = new spip::ContainerRingRead (in_db);
  input->register_buffers();

  // transfer host to device
  d_input = new spip::ContainerCUDADevice ();

  if (verbose)
    cerr << "spip::PreprocessingPipeline::configure allocating output Ring Buffer" << endl;
  d_output = new spip::ContainerRingWriteCUDA (out_db);

  if (verbose)
    cerr << "spip::PreprocessingPipeline::configure allocating d_cal_output" << endl;
  d_cal_output = new spip::ContainerCUDADevice ();

  if (verbose)
    cerr << "spip::PreprocessingPipeline::configure allocating d_trans_output" << endl;
  d_trans_output = new spip::ContainerCUDADevice ();

  if (verbose)
    cerr << "spip::PreprocessingPipeline::configure allocating cal_output" << endl;
  cal_output = new spip::ContainerBufferedRingWrite (cal_db);
  cal_output->register_buffers();

  // TODO check if Buffered, perhaps not?
  if (verbose)
    cerr << "spip::PreprocessingPipeline::configure allocating trans_output" << endl;
  trans_output = new spip::ContainerRingWrite (trans_db);
  trans_output->register_buffers();

  if (verbose)
    cerr << "spip::PreprocessingPipeline::configure_cuda allocating RAM to CUDA Transfer" << endl;
  ram_to_cuda = new spip::RAMtoCUDATransfer (stream);
  ram_to_cuda->set_input (input);
  ram_to_cuda->set_output (d_input); 
  ram_to_cuda->set_verbose (verbose);

  if (verbose)
    cerr << "spip::PreprocessingPipeline::configure_cuda unpacked container" << endl;
  // unpacked container
  unpacked = new spip::ContainerCUDADevice ();

  // unpack to float operation
  if (verbose)
    cerr << "spip::PreprocessingPipeline::configure_cuda configuring UnpackFloat" << endl;
  unpack_float = unpacker;
  unpack_float->set_input (d_input);
  unpack_float->set_output (unpacked);
  unpack_float->set_verbose (verbose);
  UnpackFloatCUDA * tmp = dynamic_cast<UnpackFloatCUDA *>(unpacker);
  tmp->set_stream (stream);

  // if no filtering is required, remove excess polarisations
  if (!filter)
  {
    if (verbose)
      cerr << "spip::PreprocessingPipeline::configure_cuda configuring PolSelectCUDA" << endl;
    // Poln Select
    pol_sel = new spip::PolSelectCUDA(stream);
    if (ref_pol > 0)
      pol_sel->set_pol_reduction ();
    pol_sel->set_input (unpacked);
    pol_sel->set_output (d_output);
    pol_sel->set_verbose (verbose);

    // if calibration is required, forward fft, integrate and write to cal_output
    if (calibrate || transients)
    {
      channelised = new spip::ContainerCUDADevice ();

      if (verbose)
        cerr << "spip::PreprocessingPipeline::configure_cuda configuring ForwardFFTCUDA" << endl;
      fwd_fft = new spip::ForwardFFTCUDA(stream);
      fwd_fft->set_input (d_output);
      fwd_fft->set_output (channelised);
      fwd_fft->set_nfft (nfft);
      fwd_fft->set_verbose (verbose);

      detected = new spip::ContainerCUDADevice();

      if (verbose)
        cerr << "spip::PreprocessingPipeline::configure_cuda configuring DetectionSquareLawCUDA" << endl;
      detect = new spip::DetectionSquareLawCUDA (stream);
      detect->set_input (channelised);
      detect->set_output (detected);
      detect->set_output_state (spip::Signal::Intensity);
      detect->set_verbose (verbose);
    }

    if (calibrate)
    {
      if (verbose)
        cerr << "spip::PreprocessingPipeline::configure_cuda configuring IntegrationBinnedCUDA" << endl;
      integrate_cal = new spip::IntegrationBinnedCUDA(stream);
      integrate_cal->set_input (detected);
      integrate_cal->set_output (d_cal_output);
      integrate_cal->set_decimation (cal_dat_dec, cal_pol_dec, cal_chan_dec, cal_signal_dec);
      integrate_cal->set_verbose (verbose);

      if (verbose)
        cerr << "spip::PreprocessingPipeline::configure_cuda configuring calibration CUDAtoRAMTransfer" << endl;
      cuda_to_ram_cal = new spip::CUDAtoRAMTransfer (stream);
      cuda_to_ram_cal->set_input (d_cal_output);
      cuda_to_ram_cal->set_output (cal_output);
      cuda_to_ram_cal->set_verbose (verbose);
    }

    if (transients)
    {
      if (verbose)
        cerr << "spip::PreprocessingPipeline::configure_cuda configuring IntegrationCUDA" << endl;
      integrate_trans = new spip::IntegrationCUDA(stream);
      integrate_trans->set_input (detected);
      integrate_trans->set_output (d_trans_output);
      integrate_trans->set_decimation (trans_dat_dec, trans_pol_dec, trans_chan_dec, trans_signal_dec);
      integrate_trans->set_verbose (verbose);

      if (verbose)
        cerr << "spip::PreprocessingPipeline::configure_cuda configuring transients CUDAtoRAMTransfer" << endl;
      cuda_to_ram_trans = new spip::CUDAtoRAMTransfer (stream);
      cuda_to_ram_trans->set_input (d_trans_output);
      cuda_to_ram_trans->set_output (trans_output);
      cuda_to_ram_trans->set_verbose (verbose);
    }
  }
  else
  {
    channelised = new spip::ContainerCUDADevice ();

    // forward FFT operation
    if (verbose)
      cerr << "spip::PreprocessingPipeline::configure_cuda configuring ForwardFFTCUDA" << endl;
    fwd_fft = new spip::ForwardFFTCUDA(stream);
    fwd_fft->set_input (unpacked);
    fwd_fft->set_output (channelised);
    fwd_fft->set_nfft (nfft);
    fwd_fft->set_verbose (verbose);

    // cleaned data
    filtered = new spip::ContainerCUDADevice ();

    string output_dir = string(".");

    // The filtering pipeline will filter pols 1+2 against pol3
    if (verbose)
      cerr << "spip::PreprocessingPipeline::configure_cuda configuring AdaptiveFilterCUDA" << endl;
    adap_fil = new spip::AdaptiveFilterCUDA(stream, output_dir);
    adap_fil->set_input (channelised);
    adap_fil->set_output (filtered);
    adap_fil->set_filtering (ref_pol);
    adap_fil->set_verbose (verbose);

    if (calibrate || transients)
    {
      detected = new spip::ContainerCUDADevice();

      if (verbose)
        cerr << "spip::PreprocessingPipeline::configure_cuda configuring DetectionSquareLawCUDA" << endl;
      detect = new spip::DetectionSquareLawCUDA (stream);
      detect->set_input (filtered);
      detect->set_output (detected);
      detect->set_output_state (spip::Signal::Intensity);
      detect->set_verbose (verbose);
    }

    if (calibrate)
    {
      if (verbose)
        cerr << "spip::PreprocessingPipeline::configure_cuda configuring IntegrationBinnedCUDA" << endl;
      integrate_cal = new spip::IntegrationBinnedCUDA(stream);
      integrate_cal->set_input (detected);
      integrate_cal->set_output (d_cal_output);
      integrate_cal->set_decimation (cal_dat_dec, cal_pol_dec, cal_chan_dec, cal_signal_dec);
      integrate_cal->set_verbose (verbose);

      // transfer device to host
      if (verbose)
        cerr << "spip::PreprocessingPipeline::configure_cuda configuring calibration CUDAtoRAMTransfer" << endl;
      cuda_to_ram_cal = new spip::CUDAtoRAMTransfer (stream);
      cuda_to_ram_cal->set_input (d_cal_output);
      cuda_to_ram_cal->set_output (cal_output);
      cuda_to_ram_cal->set_verbose (verbose);
    }

    if (transients)
    {
      if (verbose)
        cerr << "spip::PreprocessingPipeline::configure_cuda configuring IntegrationCUDA" << endl;
      integrate_trans = new spip::IntegrationCUDA(stream);
      integrate_trans->set_input (detected);
      integrate_trans->set_output (d_trans_output);
      integrate_trans->set_decimation (trans_dat_dec, trans_pol_dec, trans_chan_dec, trans_signal_dec);
      integrate_trans->set_verbose (verbose);

      // transfer device to host
      if (verbose)
        cerr << "spip::PreprocessingPipeline::configure_cuda configuring transients CUDAtoRAMTransfer" << endl;
      cuda_to_ram_trans = new spip::CUDAtoRAMTransfer (stream);
      cuda_to_ram_trans->set_input (d_trans_output);
      cuda_to_ram_trans->set_output (trans_output);
      cuda_to_ram_trans->set_verbose (verbose);
    }


    if (verbose)
      cerr << "spip::PreprocessingPipeline::configure_cuda configuring BackwardFFTCUDA" << endl;
    bwd_fft = new spip::BackwardFFTCUDA(stream);
    bwd_fft->set_input (filtered);
    bwd_fft->set_output (d_output);
    bwd_fft->set_nfft (nfft);
    bwd_fft->set_verbose (verbose);

  }
}

#endif

//! process meta-data through the pipeline, performing all resource allocation
void spip::PreprocessingPipeline::open ()
{
  if (verbose)
    cerr << "spip::PreprocessingPipeline::open()" << endl;

  if (verbose)
    cerr << "spip::PreprocessingPipeline::open input->read_header()" << endl;
  // read from the input
  input->process_header();

#ifdef HAVE_CUDA
  if (device >= 0)
  {
    if (verbose)
      cerr << "spip::PreprocessingPipeline::open ram_to_cuda->configure()" << endl;
    ram_to_cuda->configure(spip::Ordering::Custom);
  }
#endif
  
  // configure the unpacker
  if (verbose)
    cerr << "spip::PreprocessingPipeline::open unpack_float->configure(SFPT)" << endl;
  unpack_float->set_scale (1.0f / 100.0f);
  unpack_float->configure(spip::Ordering::SFPT);
 
  if (!filter)
  {
    if (verbose)
      cerr << "spip::PreprocessingPipeline::open pol_sel->configure(SFPT)" << endl;
    pol_sel->configure (spip::Ordering::SFPT);
  }

  // configure the forward fft
  if (filter || calibrate || transients)
  {
    if (verbose)
      cerr << "spip::PreprocessingPipeline::open fwd_fft->configure(SFPT)" << endl;
    fwd_fft->configure (spip::Ordering::SFPT);
  }

  if (filter) 
  { 
    if (verbose)
      cerr << "spip::PreprocessingPipeline::open adap_fil->configure(SPFT)" << endl;
    adap_fil->configure(spip::Ordering::SFPT);
  }
  
  // configure calibration and transients parts
  if (calibrate || transients)
  {
    if (verbose)
      cerr << "spip::PreprocessingPipeline::open detect->configure(TSPF)" << endl;
    detect->configure(spip::Ordering::TSPF);

    if (calibrate)
    {
      if (verbose)
        cerr << "spip::PreprocessingPipeline::open integrate_cal->configure(TSPFB)" << endl;
      integrate_cal->configure(spip::Ordering::TSPFB);

#ifdef HAVE_CUDA
      if (device >= 0)
      {
        if (verbose)
          cerr << "spip::PreprocessingPipeline::open cuda_to_ram_cal->configure(TSPFB)" << endl;
        cuda_to_ram_cal->configure(spip::Ordering::TSPFB);
      }
#endif
      cal_output->process_header();
    }

    if (calibrate)
    {
      if (verbose)
        cerr << "spip::PreprocessingPipeline::open integrate_trans->configure(TSPFB)" << endl;
      integrate_trans->configure(spip::Ordering::TSPFB);

#ifdef HAVE_CUDA
      if (device >= 0)
      {
        if (verbose)
          cerr << "spip::PreprocessingPipeline::open cuda_to_ram_trans->configure(TSPFB)" << endl;
        cuda_to_ram_trans->configure(spip::Ordering::TSPFB);
      }
#endif
      trans_output->process_header();
    }
  }

  if (filter)
  {
    if (verbose)
      cerr << "spip::PreprocessingPipeline::open bwd_fft->configure(SFPT)" << endl;
    bwd_fft->configure (spip::Ordering::SFPT);
  }

  // write the output header
#ifdef HAVE_CUDA
  if (device >= 0)
  {
    if (verbose)
      cerr << "spip::PreprocessingPipeline::open output->write_header()" << endl;
    d_output->process_header();
  }
  else
#endif
  {
    output->process_header();
  }
}

//! close the input and output data blocks
void spip::PreprocessingPipeline::close ()
{
  if (verbose)
    cerr << "spip::PreprocessingPipeline::close()" << endl;

  if (in_db->is_block_open())
  {
    if (verbose)
      cerr << "spip::PreprocessingPipeline::close in_db->close_block(" << in_db->get_data_bufsz() << ")" << endl;
    in_db->close_block (in_db->get_data_bufsz());
  }

  // close the data blocks, ending the observation
  if (verbose)
    cerr << "spip::PreprocessingPipeline::close in_db->close()" << endl;
  in_db->close();

  if (calibrate)
  {
    if (verbose)
      cerr << "spip::PreprocessingPipeline::close cal_db->close()" << endl;
    cal_db->close();
  }

  if (transients)
  {
    if (verbose)
      cerr << "spip::PreprocessingPipeline::close trans_db->close()" << endl;
    trans_db->close();
  }

  if (verbose)
    cerr << "spip::PreprocessingPipeline::close out_db->close()" << endl;
  out_db->close();

  if (verbose)
    cerr << "spip::PreprocessingPipeline::close done" << endl;
}

// process blocks of input data until the end of the data stream
bool spip::PreprocessingPipeline::process ()
{
  if (verbose)
    cerr << "spip::PreprocessingPipeline::process ()" << endl;

  bool keep_processing = true;

  // commence observation on output data block
  if (verbose)
    cerr << "spip::PreprocessingPipeline::out_db->open()" << endl;
  out_db->open();

  if (calibrate)
  {
    if (verbose)
      cerr << "spip::PreprocessingPipeline::cal_db->open()" << endl;
    cal_db->open();
  }

  if (transients)
  {
    if (verbose)
      cerr << "spip::PreprocessingPipeline::trans_db->open()" << endl;
    trans_db->open();
  }

  uint64_t input_bufsz = in_db->get_data_bufsz();
  uint64_t nbytes_input, nbytes_output;

  while (keep_processing)
  {
    // read a block of input data
    if (verbose)
      cerr << "spip::PreprocessingPipeline::process input->open_block()" << endl;
    nbytes_input = input->open_block();

    if (nbytes_input < input_bufsz)
      keep_processing = false;

    // only process full blocks of data
    if (keep_processing)
    {
#ifdef HAVE_CUDA
      if (device >= 0)
      {
        nbytes_output = d_output->open_block();
        ram_to_cuda->prepare();
        ram_to_cuda->transformation();
      }
      else
#endif
      {
        output->open_block();
      }

      if (transients)
        trans_output->open_block();

      if (verbose)
        cerr << "spip::PreprocessingPipeline::process unpack_float->transformation()" << endl;
      unpack_float->prepare();
      unpack_float->transformation();

      if (!filter)
      {
        if (verbose)
          cerr << "spip::PreprocessingPipeline::process pol_sel->prepare()" << endl;
        pol_sel->prepare();
        if (verbose)
          cerr << "spip::PreprocessingPipeline::process pol_sel->transformation()" << endl;
        pol_sel->transformation();

        if (calibrate || transients)
        {
          if (verbose)
            cerr << "spip::PreprocessingPipeline::process fwd_fft->transformation()" << endl;
          fwd_fft->prepare();
          fwd_fft->transformation();

          if (verbose)
            cerr << "spip::PreprocessingPipeline::process detect->transformation()" << endl;
          detect->prepare();
          detect->transformation ();

          if (calibrate)
          {
            if (verbose)
              cerr << "spip::PreprocessingPipeline::process integrate_cal->transformation()" << endl;
            integrate_cal->prepare();
            integrate_cal->transformation ();

#ifdef HAVE_CUDA
            if (device >= 0)
            {
              if (verbose)
                cerr << "spip::PreprocessingPipeline::process cuda_to_ram_cal->transformation()" << endl;
              cuda_to_ram_cal->prepare();
              cuda_to_ram_cal->transformation();
            }
#endif
            if (verbose)
              cerr << "spip::PreprocessingPipeline::process cal_output->write_buffer()" << endl;
            cal_output->write_buffer();
          }

          if (transients)
          {
            if (verbose)
              cerr << "spip::PreprocessingPipeline::process trans_output->open_block()" << endl;
            trans_output->open_block();

            if (verbose)
              cerr << "spip::PreprocessingPipeline::process integrate_trans->transformation()" << endl;
            integrate_trans->prepare();
            integrate_trans->transformation ();

#ifdef HAVE_CUDA
            if (device >= 0)
            {
              if (verbose)
                cerr << "spip::PreprocessingPipeline::process cuda_to_ram_trans->transformation()" << endl;
              cuda_to_ram_trans->prepare();
              cuda_to_ram_trans->transformation();
            }
#endif
            if (verbose)
              cerr << "spip::PreprocessingPipeline::process trans_output->close_block()" << endl;
            trans_output->close_block();
          }
        }
      }
      else
      {
        fwd_fft->prepare();
        fwd_fft->transformation();

        adap_fil->prepare();
        adap_fil->transformation();

        bwd_fft->prepare();
        bwd_fft->transformation();

        if (calibrate || transients)
        {
          fwd_fft->prepare();
          fwd_fft->transformation();

          detect->prepare();
          detect->transformation ();
        }

        if (calibrate)
        {
          integrate_cal->prepare();
          integrate_cal->transformation ();

#ifdef HAVE_CUDA
          if (device >= 0)
          {
            if (verbose)
              cerr << "spip::PreprocessingPipeline::process cuda_to_ram_cal->transformation()" << endl;
            cuda_to_ram_cal->prepare();
            cuda_to_ram_cal->transformation();
          }
#endif
          if (verbose)
            cerr << "spip::PreprocessingPipeline::process cal_output->write_buffer()" << endl;
          cal_output->write_buffer();
        }

        if (transients)
        {
          if (verbose)
            cerr << "spip::PreprocessingPipeline::process trans_output->open_block()" << endl;
          trans_output->open_block();

          integrate_trans->prepare();
          integrate_trans->transformation ();

#ifdef HAVE_CUDA
          if (device >= 0)
          {
            if (verbose)
              cerr << "spip::PreprocessingPipeline::process cuda_to_ram_trans->transformation()" << endl;
            cuda_to_ram_trans->prepare();
            cuda_to_ram_trans->transformation();
          }
#endif
          if (verbose)
            cerr << "spip::PreprocessingPipeline::process trans_output->close_block()" << endl;
          trans_output->close_block();
        }
      }
#ifdef HAVE_CUDA
      if (device >= 0)
      {
        if (verbose)
          cerr << "spip::PreprocessingPipeline::process d_output->close_block()" << endl;
        d_output->close_block();
      }
      else
#endif
      {
        if (verbose)
          cerr << "spip::PreprocessingPipeline::process output->close_block()" << endl;
        output->close_block();
      }
    }

    if (verbose)
      cerr << "spip::PreprocessingPipeline::process input->close_block()" << endl;
    input->close_block();
  }

  if (verbose)
    cerr << "spip::PreprocessingPipeline::process return true" << endl;
  return true;
}
