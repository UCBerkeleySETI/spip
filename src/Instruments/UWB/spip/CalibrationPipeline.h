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
#include "spip/SampleFoldRAM.h"
#include "spip/ContainerRingRead.h"
#include "spip/ContainerBufferedRingWrite.h"
#include "spip/ContainerRAM.h"

#include "config.h"

#ifdef HAVE_CUDA
#include "spip/ContainerCUDA.h"
#include "spip/UnpackFloatCUDA.h"
#include "spip/SampleFoldCUDA.h"
#include "spip/RAMtoCUDATransfer.h"
#include "spip/CUDAtoRAMTransfer.h"
#endif

#include <vector>

namespace spip {

  class CalibrationPipeline {

    public:

      CalibrationPipeline (const char * in_key_string, const char * out_key_string);

      ~CalibrationPipeline ();

      void set_periodicity (unsigned, uint64_t);

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

      SampleFold * sample_fold;

      ContainerRingRead * input;

      Container * unpacked;

      ContainerBufferedRingWrite * output;

      Signal::State output_state;

      unsigned nbin;

      uint64_t dat_offset;

      uint64_t dat_dec;

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
