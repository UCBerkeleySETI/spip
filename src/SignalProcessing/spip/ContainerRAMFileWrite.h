/***************************************************************************
 *
 *   Copyright (C) 2016 Andrew Jameson
 *   Licensed under the Academic Free License version 2.1
 *
 ***************************************************************************/

#include "config.h"
#include "spip/ContainerRAM.h"
#include "spip/FileWrite.h"

#ifndef __ContainerRAMFileWrite_h
#define __ContainerRAMFileWrite_h

namespace spip {

  class ContainerRAMFileWrite : public ContainerRAM, public FileWrite
  {
    public:

      //! Null constructor
      ContainerRAMFileWrite (std::string);

      ~ContainerRAMFileWrite();

      //! process the input header and configure
      void process_header ();

      //! write header to the current file
      void write_header ();

      //! write a total of ndat to the file
      void write (uint64_t ndat);

      //! write data to the current file
      uint64_t write_data (uint64_t start_idat, uint64_t ndat_to_write);

    protected:

    private:

  };
}

#endif
