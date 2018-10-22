/***************************************************************************
 *
 *   Copyright (C) 2018 Andrew Jameson
 *   Licensed under the Academic Free License version 2.1
 *
 ***************************************************************************/

#ifndef __PreprocessingPipeline_h
#define __PreprocessingPipeline_h

#include "spip/AsciiHeader.h"
#include "spip/Time.h"
#include "spip/DataBlockRead.h"
#include "spip/DataBlockWrite.h"
#include "spip/UnpackFloatRAM.h"
#include "spip/ForwardFFTFFTW.h"
#include "spip/AdaptiveFilterRAM.h"
#include "spip/BackwardFFTFFTW.h"
#include "spip/DetectionSquareLawRAM.h"
#include "spip/IntegrationBinnedRAM.h"
#include "spip/IntegrationRAM.h"
#include "spip/PolSelectRAM.h"
#include "spip/ContainerRingRead.h"
#include "spip/ContainerBufferedRingWrite.h"
#include "spip/ContainerRingWrite.h"
#include "spip/ContainerRAM.h"

#include "config.h"

#ifdef HAVE_CUDA
#include "spip/ContainerCUDA.h"
#include "spip/UnpackFloatCUDA.h"
#include "spip/ForwardFFTCUDA.h"
#include "spip/AdaptiveFilterCUDA.h"
#include "spip/BackwardFFTCUDA.h"
#include "spip/DetectionSquareLawCUDA.h"
#include "spip/IntegrationBinnedCUDA.h"
#include "spip/IntegrationCUDA.h"
#include "spip/PolSelectCUDA.h"
#include "spip/RAMtoCUDATransfer.h"
#include "spip/CUDAtoRAMTransfer.h"
#include "spip/ContainerRingWriteCUDA.h"
#endif

#include <vector>

namespace spip {

  class PreprocessingPipeline {

    public:

      PreprocessingPipeline (const char *, const char *, const char *, const char *);

      ~PreprocessingPipeline ();

      void set_function (bool, bool, bool);

      void set_filtering (int);

      void set_cal_decimation (unsigned, uint64_t, unsigned);

      void set_trans_decimation (uint64_t, unsigned);

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

      DataBlockWrite * cal_db;

      DataBlockWrite * trans_db;

      DataBlockWrite * out_db;

      UnpackFloat * unpack_float;

      PolSelect * pol_sel;

      ForwardFFT * fwd_fft;

      AdaptiveFilter * adap_fil;

      BackwardFFT * bwd_fft;

      DetectionSquareLaw * detect;

      IntegrationBinned * integrate_cal;

      Integration * integrate_trans;

      ContainerRingRead * input;

      Container * unpacked;

      Container * channelised;

      Container * detected;

      Container * filtered;

      ContainerBufferedRingWrite * cal_output;

      ContainerRingWrite * trans_output;

      ContainerRingWrite * output;

#ifdef HAVE_CUDA
      ContainerRingWriteCUDA * d_output;
#endif

      Signal::State output_state;

      Signal::State cal_output_state;

      Signal::State trans_output_state;

      unsigned nfft;

      uint64_t cal_dat_dec;

      unsigned cal_pol_dec;

      unsigned cal_chan_dec;

      unsigned cal_signal_dec;

      uint64_t trans_dat_dec;

      unsigned trans_pol_dec;

      unsigned trans_chan_dec;

      unsigned trans_signal_dec;

      bool verbose;

#ifdef HAVE_CUDA
      int device;

      cudaStream_t stream;

      RAMtoCUDATransfer * ram_to_cuda;

      ContainerCUDADevice * d_input;

      ContainerCUDADevice * d_cal_output; 

      ContainerCUDADevice * d_trans_output; 

      CUDAtoRAMTransfer * cuda_to_ram_cal;

      CUDAtoRAMTransfer * cuda_to_ram_trans;
#endif

      bool calibrate;

      bool transients;

      bool filter;

      unsigned npol;
      
      unsigned out_npol;
      
      int ref_pol;
  };

}

#endif
