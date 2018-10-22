/***************************************************************************
 *
 *   Copyright (C) 2016 Andrew Jameson
 *   Licensed under the Academic Free License version 2.1
 *
 ***************************************************************************/

#include "config.h"

#include "spip/File.h"
#include "spip/Container.h"

#ifndef __FileWrite_h
#define __FileWrite_h

namespace spip {

  class FileWrite : public File
  {
    public:

      //! Null constructor
      FileWrite (std::string);

      ~FileWrite();

      void configure (const Container *);

      void set_filename (std::string);

      void set_filename_suffix (std::string);

      void set_file_length_seconds (float);

      void set_file_length_bytes (int64_t);

      void resize_hdr_buf (unsigned);

      void open_file (const char *, uint64_t);

      void close_file ();

      void discard_file ();

      void write_header (const char *, size_t);

      uint64_t write_data_bytes (int, void *, uint64_t, uint64_t);

      virtual uint64_t write_data (uint64_t, uint64_t) = 0;

    protected:

       //! number of data samples read from the buffer
       uint64_t idat_written;

       //! number of data samples to write in each file
       uint64_t ndat_per_file;

    private:

      bool fixed_filename;

      //! directory to which files will be written
      std::string dir;

      //! use a temporary filename to prevent premature reading of file
      std::string temporary_filename;

      //! suffix/extension to be used on written files
      std::string filename_suffix;

      //! file number to write
      uint64_t file_number;

      //! previous obs_offset
      int64_t prev_obs_offset;

      float file_length;

      char * hdr_buf;

      size_t hdr_bufsz;

      float desired_secs;

      int64_t desired_bytes;

  };
}

#endif
