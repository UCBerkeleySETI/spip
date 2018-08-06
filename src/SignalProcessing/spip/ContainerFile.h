/***************************************************************************
 *
 *   Copyright (C) 2018 Andrew Jameson
 *   Licensed under the Academic Free License version 2.1
 *
 ***************************************************************************/

#include "config.h"
#include "spip/ContainerRAM.h"

#ifndef __ContainerFile_h
#define __ContainerFile_h

namespace spip {

  class ContainerFile : public ContainerRAM
  {
    public:

      //! 
      ContainerFile ();

      ~ContainerFile();

      virtual void process_header () = 0;

      void close_file ();

    protected:

      // name of the file
      std::string filename;

      // file descriptor 
      int fd;

      // size of the header to be read/written in bytes
      unsigned hdr_size;

      // version number of the header to be read/written in bytes
      unsigned hdr_version;

    private:

  };
}

#endif
