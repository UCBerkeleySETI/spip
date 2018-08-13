/***************************************************************************
 *
 *   Copyright (C) 2018 Andrew Jameson
 *   Licensed under the Academic Free License version 2.1
 *
 ***************************************************************************/

#ifndef __CalibrationPipeline_h
#define __CalibrationPipeline_h

#include "spip/AsciiHeader.h"
#include "spip/Time.h"
#include "spip/DataBlockRead.h"
#include "spip/DataBlockWrite.h"
#include "spip/UnpackFloatRAM.h"
#include "spip/ForwardFFTFFTW.h"
#include "spip/DetectionSquareLawRAM.h"
#include "spip/IntegrationBinnedRAM.h"
#include "spip/ContainerRingRead.h"
#include "spip/ContainerBufferedRingWrite.h"
#include "spip/ContainerRAM.h"

#include "config.h"

#ifdef HAVE_CUDA
#include "spip/ContainerCUDA.h"
#include "spip/UnpackFloatCUDA.h"
#include "spip/ForwardFFTCUDA.h"
#include "spip/DetectionSquareLawCUDA.h"
#include "spip/IntegrationBinnedCUDA.h"
#include "spip/RAMtoCUDATransfer.h"
#include "spip/CUDAtoRAMTransfer.h"
#endif

#include <vector>

namespace spip {

  class CalibrationPipeline {

    public:

      CalibrationPipeline (const char * in_key_string, const char * out_key_string);

      ~CalibrationPipeline ();

      void set_decimation (uint64_t, unsigned, unsigned);

      void set_channelisation (unsigned);

      void set_output_state (Signal::State);

      void configure (UnpackFloat *);

#ifdef HAVE_CUDA
      void set_device (int _device);

      void configure_cuda (UnpackFloat *);
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

      UnpackFloat * unpack_float;

      ForwardFFT * fwd_fft;

      DetectionSquareLaw * detection;

      IntegrationBinned * integration_binned;

      ContainerRingRead * input;

      Container * unpacked;

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

      RAMtoCUDATransfer * ram_to_cuda; 

      ContainerCUDADevice * d_input;

      ContainerCUDADevice * d_output; 

      CUDAtoRAMTransfer * cuda_to_ram; 
#endif
  };

}

#endif
