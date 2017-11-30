/***************************************************************************
 *
 *   Copyright (C) 2016 Andrew Jameson
 *   Licensed under the Academic Free License version 2.1
 *
 ***************************************************************************/

#ifndef __AdaptiveFilterTest_h
#define __AdaptiveFilterTest_h

#include "spip/AsciiHeader.h"
#include "spip/Time.h"
#include "spip/DataBlockRead.h"
#include "spip/DataBlockWrite.h"
#include "spip/UnpackFloatRAM.h"
#include "spip/AdaptiveFilterRAM.h"
#include "spip/ContainerRingRead.h"
#include "spip/ContainerRingWrite.h"
#include "spip/ContainerRAM.h"

#include "config.h"

#ifdef HAVE_CUDA
#include "spip/ContainerCUDA.h"
#include "spip/UnpackFloatCUDA.h"
#include "spip/AdaptiveFilterCUDA.h"
#include "spip/RAMtoCUDATransfer.h"
#include "spip/CUDAtoRAMTransfer.h"
#endif

#include <vector>

namespace spip {

  class AdaptiveFilterTest {

    public:

      AdaptiveFilterTest (const char * in_key_string, const char * ref_key_string, const char * out_key_string);

      ~AdaptiveFilterTest ();

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

      DataBlockRead * ref_db;

      DataBlockWrite * out_db;

      UnpackFloat * unpack_float;

      UnpackFloat * unpack_float_ref;

      AdaptiveFilter * filter;

      ContainerRingRead * input;

      ContainerRingRead * input_ref;

      Container * unpacked;

      Container * unpacked_ref;

      ContainerRingWrite * output;

      int nfft;

      bool verbose;

#ifdef HAVE_CUDA
      int device;

      cudaStream_t stream;

      RAMtoCUDATransfer * ram_to_cuda; 

      RAMtoCUDATransfer * ram_to_cuda_ref; 

      ContainerCUDADevice * d_input;

      ContainerCUDADevice * d_input_ref; 

      ContainerCUDADevice * d_output; 

      CUDAtoRAMTransfer * cuda_to_ram; 
#endif
  };

}

#endif
