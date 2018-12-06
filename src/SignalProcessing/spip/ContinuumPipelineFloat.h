/***************************************************************************
 *
 *   Copyright (C) 2018 Andrew Jameson
 *   Licensed under the Academic Free License version 2.1
 *
 ***************************************************************************/

#ifndef __ContinuumPipelineFloat_h
#define __ContinuumPipelineFloat_h

#include "spip/AsciiHeader.h"
#include "spip/Time.h"
#include "spip/DataBlockRead.h"
#include "spip/DataBlockWrite.h"
#include "spip/ForwardFFTFFTW.h"
#include "spip/DetectionPolarimetryRAM.h"
#include "spip/DetectionSquareLawRAM.h"
#include "spip/IntegrationRAM.h"
#include "spip/RAMtoRAMTransfer.h"
#include "spip/ReverseFrequencyRAM.h"
#include "spip/ContainerRingRead.h"
#include "spip/ContainerRAMFileWrite.h"
#include "spip/ContainerRAM.h"

#include "config.h"

#ifdef HAVE_CUDA
#include "spip/ContainerCUDA.h"
#include "spip/ContainerRingReadCUDA.h"
#include "spip/ContainerCUDAFileWrite.h"
#include "spip/ForwardFFTCUDA.h"
#include "spip/DetectionPolarimetryCUDA.h"
#include "spip/DetectionSquareLawCUDA.h"
#include "spip/IntegrationCUDA.h"
#include "spip/ReverseFrequencyCUDA.h"
#include "spip/RINGtoCUDATransfer.h"
#include "spip/CUDAtoRAMTransfer.h"
#endif

#include <vector>

namespace spip {

  class ContinuumPipelineFloat {

    public:

      ContinuumPipelineFloat (const char *, const char *);

      ~ContinuumPipelineFloat ();

      void set_channelisation (int);

      void set_channel_oversampling (int);

      void set_decimation (int);

      void set_tsubint (float);

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

      ForwardFFT * fwd_fft;

      Detection * detector;

      Integration * integrator;

      ReverseFrequency * reverser;

      ContainerRingRead * input;

      Container * reblocked;

      Container * channelised;

      Container * detected;

      Container * integrated;

      ContainerRAMFileWrite * output;

      Signal::State output_state;

      std::string out_dir;

      int nfft;

      int nchan_out;

      int channel_oversampling;

      int tdec;

      int fdec;

      float tsubint;

      bool verbose;

#ifdef HAVE_CUDA
      int device;

      cudaStream_t stream;

      ContainerCUDAFileWrite * d_output; 

      RINGtoCUDATransfer * ring_to_cuda; 
#endif

      RAMtoRAMTransfer * ram_to_ram; 

      unsigned reblock_factor;
  };

}

#endif
