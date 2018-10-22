/***************************************************************************
 *
 *   Copyright (C) 2018 Andrew Jameson
 *   Licensed under the Academic Free License version 2.1
 *
 ***************************************************************************/

#include "spip/File.h"

#ifndef __FileRead_h
#define __FileRead_h

namespace spip {

  class FileRead : public File
  {
    public:

      //! Null constructor
      FileRead (std::string);

      ~FileRead();

      void process_header();

      void open_file ();

      uint64_t read_data ();

    protected:

      //! total size of the file, header + data
      size_t file_size_total;

      //! size of the data (file - header)
      size_t data_size;

    private:

  };
}

#endif
