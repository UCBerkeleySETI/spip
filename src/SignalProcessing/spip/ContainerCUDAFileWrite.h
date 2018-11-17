/***************************************************************************
 *
 *   Copyright (C) 2018 Andrew Jameson
 *   Licensed under the Academic Free License version 2.1
 *
 ***************************************************************************/

#include "config.h"
#include "spip/ContainerCUDA.h"
#include "spip/FileWrite.h"

#ifndef __ContainerCUDAFileWrite_h
#define __ContainerCUDAFileWrite_h

namespace spip {

  class ContainerCUDAFileWrite : public ContainerCUDADevice, public FileWrite
  {
    public:

      //! Null constructor
      ContainerCUDAFileWrite (cudaStream_t, std::string);

      ~ContainerCUDAFileWrite();

      //! process the input header and configure
      void process_header ();

      //! write header to the current file
      void write_header ();

      //! write a total of ndat to the file
      void write (uint64_t ndat);

      //! write data to the current file
      uint64_t write_data (uint64_t start_idat, uint64_t ndat_to_write);

      //! resize the host buffer)
      void resize_host_buffer (size_t required_size);

    protected:

    private:

      cudaStream_t stream;

      void * host_buffer;

      size_t host_buffer_size;

  };
}

#endif
