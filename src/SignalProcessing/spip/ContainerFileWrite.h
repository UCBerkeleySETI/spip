/***************************************************************************
 *
 *   Copyright (C) 2016 Andrew Jameson
 *   Licensed under the Academic Free License version 2.1
 *
 ***************************************************************************/

#include "config.h"
#include "spip/ContainerFile.h"

#ifndef __ContainerFileWrite_h
#define __ContainerFileWrite_h

namespace spip {

  class ContainerFileWrite : public ContainerFile
  {
    public:

      //! Null constructor
      ContainerFileWrite (std::string);

      ~ContainerFileWrite();

      void process_header ();

      void set_filename (std::string);

      void set_file_length_seconds (float);

      void set_file_length_bytes (int64_t);

      void resize_hdr_buf (unsigned);

      void write ();

      void open_file ();

      void close_file ();

      void discard_file ();

      void write_header ();

      uint64_t write_data (uint64_t, uint64_t);

    protected:

    private:

      bool fixed_filename;

      //! directory to which files will be written
      std::string dir;

      //! use a temporary filename to prevent premature reading of file
      std::string temporary_filename;

      //! file number to write
      uint64_t file_number;

      uint64_t ndat_per_file;

      float file_length;

      char * hdr_buf;

      size_t hdr_bufsz;

      // number of data samples read from the buffer
      uint64_t idat_written;

      // for CUDA transfers
      bool buffer_registered;

      // for CUDA transfers
      bool buffer_was_registered;

      float desired_secs;

      int64_t desired_bytes;

  };
}

#endif
