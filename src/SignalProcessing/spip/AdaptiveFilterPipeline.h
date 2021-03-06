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

      AdaptiveFilterPipeline (const char * in_key_string, const char * out_key_string);

      ~AdaptiveFilterPipeline ();

      void set_channelisation (int freq_res);

      void configure (UnpackFloat *);

      void set_filtering (int, double);

#ifdef HAVE_CUDA
      void set_device (int _device);

      void configure_cuda(UnpackFloat *);
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

      AdaptiveFilter * filter;

      BackwardFFT * bwd_fft;

      ContainerRingRead * input;

      Container * unpacked;

      Container * channelised;

      Container * cleaned;

      ContainerRingWrite * output;

      int nfft;

      int reference_pol;

      bool verbose;

#ifdef HAVE_CUDA
      int device;

      cudaStream_t stream;

      RAMtoCUDATransfer * ram_to_cuda; 

      ContainerCUDADevice * d_input;

      ContainerCUDADevice * d_output; 

      CUDAtoRAMTransfer * cuda_to_ram; 
#endif

      double input_bytes_per_second;

      double req_mon_tsamp;

  };

}

#endif
