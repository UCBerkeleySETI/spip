/***************************************************************************
 *
 *   Copyright (C) 2016 Andrew Jameson
 *   Licensed under the Academic Free License version 2.1
 *
 ***************************************************************************/

#ifndef __AdaptiveFilterPipeline_h
#define __AdaptiveFilterPipeline_h

#include "spip/AsciiHeader.h"
#include "spip/Time.h"
#include "spip/DataBlockRead.h"
#include "spip/DataBlockWrite.h"
#include "spip/UnpackFloatRAM.h"
#include "spip/ForwardFFTFFTW.h"
#include "spip/BackwardFFTFFTW.h"
#include "spip/AdaptiveFilterRAM.h"
#include "spip/ContainerRingRead.h"
#include "spip/ContainerRingWrite.h"
#include "spip/ContainerRAM.h"

#include "config.h"

#ifdef HAVE_CUDA
#include "spip/ContainerCUDA.h"
#include "spip/UnpackFloatCUDA.h"
#include "spip/ForwardFFTCUDA.h"
#include "spip/AdaptiveFilterCUDA.h"
#include "spip/BackwardFFTCUDA.h"
#include "spip/RAMtoCUDATransfer.h"
#include "spip/CUDAtoRAMTransfer.h"
#endif

#include <vector>

namespace spip {

  class AdaptiveFilterPipeline {

    public:

      AdaptiveFilterPipeline (const char * in_key_string, const char * rfi_key_string, const char * out_key_string);

      ~AdaptiveFilterPipeline ();

      void set_channelisation (int freq_res);

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

      DataBlockRead * rfi_db;

      DataBlockWrite * out_db;

      UnpackFloat * unpack_float;

      UnpackFloat * unpack_float_rfi;

      ForwardFFT * fwd_fft;

      ForwardFFT * fwd_fft_rfi;

      AdaptiveFilter * filter;

      BackwardFFT * bwd_fft;

      ContainerRingRead * input;

      ContainerRingRead * input_rfi;

      Container * unpacked;

      Container * unpacked_rfi;

      Container * channelised;

      Container * channelised_rfi;

      Container * cleaned;

      ContainerRingWrite * output;

      int nfft;

      bool verbose;

#ifdef HAVE_CUDA
      int device;

      cudaStream_t stream;

      RAMtoCUDATransfer * ram_to_cuda; 

      RAMtoCUDATransfer * ram_to_cuda_rfi; 

      ContainerCUDADevice * d_input;

      ContainerCUDADevice * d_input_rfi; 

      ContainerCUDADevice * d_output; 

      CUDAtoRAMTransfer * cuda_to_ram; 
#endif
  };

}

#endif
