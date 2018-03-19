/***************************************************************************
 *
 *   Copyright (C) 2018 Andrew Jameson
 *   Licensed under the Academic Free License version 2.1
 *
 ***************************************************************************/

#ifndef __ContinuumPipeline_h
#define __ContinuumPipeline_h

#include "spip/AsciiHeader.h"
#include "spip/Time.h"
#include "spip/DataBlockRead.h"
#include "spip/DataBlockWrite.h"
#include "spip/UnpackFloatRAM.h"
#include "spip/ForwardFFTFFTW.h"
#include "spip/DetectionPolarimetryRAM.h"
#include "spip/DetectionSquareLawRAM.h"
#include "spip/IntegrationRAM.h"
#include "spip/ContainerRingRead.h"
#include "spip/ContainerBufferedRingWrite.h"
#include "spip/ContainerRAM.h"

#include "config.h"

#ifdef HAVE_CUDA
#include "spip/ContainerCUDA.h"
#include "spip/UnpackFloatCUDA.h"
#include "spip/ForwardFFTCUDA.h"
#include "spip/DetectionPolarimetryCUDA.h"
#include "spip/DetectionSquareLawCUDA.h"
#include "spip/IntegrationCUDA.h"
#include "spip/RAMtoCUDATransfer.h"
#include "spip/CUDAtoRAMTransfer.h"
#endif

#include <vector>

namespace spip {

  class ContinuumPipeline {

    public:

      ContinuumPipeline (const char * in_key_string, const char * out_key_string);

      ~ContinuumPipeline ();

      void set_channelisation (int);

      void set_channel_oversampling (int);

      void set_decimation (int);

      void set_output_state (Signal::State);

      void configure ();

#ifdef HAVE_CUDA
      void set_device (int _device);

      void configure_cuda();
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

      Detection * detector;

      Integration * integrator;

      ContainerRingRead * input;

      Container * unpacked;

      Container * channelised;

      Container * detected;

      Container * integrated;

      ContainerBufferedRingWrite * output;

      Signal::State output_state;

      int nfft;

      int nchan_out;

      int channel_oversampling;

      int tdec;

      int fdec;

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
