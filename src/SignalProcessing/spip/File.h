/***************************************************************************
 *
 *   Copyright (C) 2018 Andrew Jameson
 *   Licensed under the Academic Free License version 2.1
 *
 ***************************************************************************/

#include "config.h"

#include <string>

#ifndef __File_h
#define __File_h

namespace spip {

  class File 
  {
    public:

      static bool verbose;

      //! 
      File ();

      ~File();

      virtual void process_header () = 0;

      void close_file ();

    protected:

      // name of the file
      std::string filename;

      // file descriptor 
      int fd;

    private:

  };
}

#endif
