/***************************************************************************
 *
 *   Copyright (C) 2018 Andrew Jameson
 *   Licensed under the Academic Free License version 2.1
 *
 ***************************************************************************/

#include "spip/FileWrite.h"
#include "spip/AsciiHeader.h"
#include "spip/Error.h"

#include <iostream>
#include <cstring>

#include <unistd.h>
#include <fcntl.h>

using namespace std;

spip::FileWrite::FileWrite (std::string _dir)
{
  dir = _dir;
  file_number = 0;
  fd = -1;
  ndat_per_file = -1; // by default write 10s to file
  file_length = 10;   // seconds

  hdr_buf = NULL;
  hdr_bufsz = 0;

  desired_secs = -1;
  desired_bytes = -1;
  desired_bytes = -1;

  filename_suffix = "dada";
  fixed_filename = false;
  
  file_number = 0;
  prev_obs_offset = -1;
}

spip::FileWrite::~FileWrite ()
{
  // if the file is still open
  if (fd > -1)
  {
    // if the required amount of data has been written to the file, close it
    if (idat_written >= ndat_per_file)
      close_file();
    else
      discard_file();
  }
}

//! write data stored in the buffer to disk
void spip::FileWrite::configure (const spip::Container * input)
{
  double bytes_per_second = input->calculate_bytes_per_second ();
  unsigned bits_per_sample = input->calculate_nbits_per_sample ();
  int64_t file_size = input->get_file_size ();

  if (spip::File::verbose)
    cerr << "spip::FileWrite::configure desired_secs=" << desired_secs 
         << " desired_bytes=" << desired_bytes << " bytes_per_second=" 
         << bytes_per_second << " file_size=" << file_size << endl;

  uint64_t bytes;
  if (desired_secs >= 0)
  {
    bytes = uint64_t (bytes_per_second * desired_secs);
  }
  else if (desired_bytes >= 0)
  {
    bytes = uint64_t(desired_bytes);
  }
  else if (file_size == -1)
  {
    bytes = uint64_t (bytes_per_second * file_length);
  }
  else
  {
    bytes = file_size;
  }

  ndat_per_file = (bytes * 8) / bits_per_sample;
  if (spip::File::verbose)
    cerr << "spip::FileWrite::configure desired_secs=" << desired_secs 
         << " bytes= " << bytes << " bits_per_sample=" << bits_per_sample 
         << " ndat_per_file=" << ndat_per_file << endl;

  // number of dat processed from the current buffer
  idat_written = 0;
}

void spip::FileWrite::set_filename (std::string _filename)
{
  filename = _filename;
  fixed_filename = true;
}

void spip::FileWrite::set_filename_suffix (std::string suffix)
{
  filename_suffix = suffix;
}

void spip::FileWrite::set_file_length_bytes (int64_t bytes)
{
  desired_bytes = bytes;
  if (spip::File::verbose)
    cerr << "spip::FileWrite::set_file_length_bytes bytes=" << bytes << endl;
}

void spip::FileWrite::set_file_length_seconds (float secs)
{
  desired_secs = secs;
  if (spip::File::verbose)
    cerr << "spip::FileWrite::set_file_length_seconds desired_secs=" << secs << endl;
}

void spip::FileWrite::resize_hdr_buf (unsigned required_size)
{
  if (spip::File::verbose)
    cerr << "spip::FileWrite::resize_hdr_buf required_size=" << required_size << endl;
  if (required_size > hdr_bufsz)
  {
    if (spip::File::verbose)
      cerr << "spip::FileWrite::resize_hdr_buf resizing from " << hdr_bufsz << endl;
    if (hdr_buf) 
      free (hdr_buf);
    hdr_buf = (char *) malloc (required_size);
    hdr_bufsz = required_size;
  }
}

void spip::FileWrite::write_header (const char * hdr, size_t hdr_size)
{
  // ensure the local buffer to be large enough for the full header
  resize_hdr_buf (hdr_size);
  
  // zero the header buffer
  memset (hdr_buf, '\0', hdr_size);

  strncpy (hdr_buf, hdr, strlen(hdr));

  // write the full header to the file
  ::write (fd, hdr_buf, hdr_size);
}

void spip::FileWrite::open_file (const char * utc_start_str, uint64_t obs_offset)
{
  if (fd >= 0)
    throw Error(InvalidState, "spip::FileWrite::open_file", "file descriptor already open");

  int flags = O_WRONLY | O_CREAT | O_TRUNC;
  int perms = S_IRUSR | S_IWUSR | S_IRGRP | S_IROTH;

  if (!fixed_filename)
  {
    // incrememnt file_number if obs_offset matching
    if (int64_t(obs_offset) == prev_obs_offset)
      file_number++;
    else
      file_number = 0;
    prev_obs_offset = int64_t(obs_offset);

    char filename_buf[FILENAME_MAX];
    snprintf (filename_buf, FILENAME_MAX, "%s_%016" PRIu64 ".%06lu.%s", 
              utc_start_str, obs_offset, file_number, filename_suffix.c_str());
    filename = dir + "/" + string(filename_buf);
    temporary_filename = dir + "/." + string(filename_buf) + ".tmp";
    if (verbose)
      cerr << "Creating " << filename_buf << endl;
  }
  else
  {
    temporary_filename = filename + ".tmp";
  }

  fd = open (temporary_filename.c_str(), flags, perms);
}

uint64_t spip::FileWrite::write_data_bytes (int fd, void * data, uint64_t bytes_to_write, uint64_t ndat)
{
  uint64_t bytes_written = ::write (fd, data, bytes_to_write);
  if (bytes_written != bytes_to_write)
    throw Error(InvalidState, "spip::FileWrite::write_data_bytes", 
                "bytes_to_write=%lu but bytes_written=%lu",
                bytes_to_write, bytes_written);
  idat_written += ndat;
  return bytes_written;
}


void spip::FileWrite::close_file()
{
  if (spip::File::verbose)
    cerr << "spip::FileWrite::close_file" << endl;
  spip::File::close_file();

  // rename temporary file to actual filename
  rename (temporary_filename.c_str(), filename.c_str());
}

void spip::FileWrite::discard_file()
{
  if (spip::File::verbose)
    cerr << "spip::FileWrite::discard_file" << endl;
  spip::File::close_file();

  // delete the temporary file
  remove(temporary_filename.c_str());
}

