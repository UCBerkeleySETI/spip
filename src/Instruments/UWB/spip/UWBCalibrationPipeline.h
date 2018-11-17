/***************************************************************************
 *
 *   Copyright (C) 2018 Andrew Jameson
 *   Licensed under the Academic Free License version 2.1
 *
 ***************************************************************************/

#ifndef __UWBCalibrationPipeline_h
#define __UWBCalibrationPipeline_h

#include "spip/AsciiHeader.h"
#include "spip/Time.h"
#include "spip/DataBlockRead.h"
#include "spip/DataBlockWrite.h"
#include "spip/ForwardFFTFFTW.h"
#include "spip/DetectionSquareLawRAM.h"
#include "spip/IntegrationBinnedRAM.h"
#include "spip/ContainerRingRead.h"
#include "spip/ContainerBufferedRingWrite.h"
#include "spip/ContainerRAM.h"

#include "config.h"

#ifdef HAVE_CUDA
#include "spip/ContainerCUDA.h"
#include "spip/ContainerRingReadCUDA.h"
#include "spip/ForwardFFTCUDA.h"
#include "spip/DetectionSquareLawCUDA.h"
#include "spip/IntegrationBinnedCUDA.h"
#include "spip/CUDAtoRAMTransfer.h"
#endif

#include <vector>

namespace spip {

  class UWBCalibrationPipeline {

    public:

      UWBCalibrationPipeline (const char * in_key_string, const char * out_key_string);

      ~UWBCalibrationPipeline ();

      void set_decimation (uint64_t, unsigned, unsigned);

      void set_channelisation (unsigned);

      void set_output_state (Signal::State);

      void configure ();

#ifdef HAVE_CUDA
      void set_device (int _device);

      void configure_cuda ();
#endif

      void open ();

      void open (const char * header_str);

      void close ();

      bool process ();

      void set_verbose () { verbose = true; };

    private:

      AsciiHeader header;

      DataBlockRead * in_db;

      DataBlockWrite * out_db;

      ForwardFFT * fwd_fft;

      DetectionSquareLaw * detection;

      IntegrationBinned * integration_binned;

      ContainerRingRead * input;

      Container * channelised;

      Container * detected;

      ContainerBufferedRingWrite * output;

      Signal::State output_state;

      unsigned nfft;

      uint64_t dat_dec;

      unsigned pol_dec;

      unsigned chan_dec;

      unsigned signal_dec;

      bool verbose;

#ifdef HAVE_CUDA
      int device;

      cudaStream_t stream;

      ContainerRingReadCUDA * d_input;

      ContainerCUDADevice * d_output; 

      CUDAtoRAMTransfer * cuda_to_ram; 
#endif
  };

}

#endif
