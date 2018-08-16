/***************************************************************************
 *
 *   Copyright (C) 2018 Andrew Jameson
 *   Licensed under the Academic Free License version 2.1
 *
 ***************************************************************************/

#include "spip/ContainerFileWrite.h"
#include "spip/AsciiHeader.h"
#include "spip/Error.h"

#include <iostream>
#include <cstring>

#include <unistd.h>
#include <fcntl.h>

using namespace std;

spip::ContainerFileWrite::ContainerFileWrite (std::string _dir)
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

  fixed_filename = false;
}

spip::ContainerFileWrite::~ContainerFileWrite ()
{
  // if the file is still open, write out the remainder
  if (fd > -1)
    close_file();
}

//! write data stored in the buffer to disk
void spip::ContainerFileWrite::process_header ()
{
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
  if (verbose)
    cerr << "spip::ContainerFileWrite::process_header desired_secs=" << desired_secs 
         << " bytes= " << bytes << " bits_per_sample=" << bits_per_sample 
         << " ndat_per_file=" << ndat_per_file << endl;

  // read required header parameters
  if (header.get ("HDR_SIZE", "%u", &hdr_size) != 1)
    hdr_size = 4096;
  if (header.get ("HDR_VERSION", "%f", &hdr_version) != 1)
    hdr_version = 1.0;

  std::string utc_start_str = utc_start->get_gmtime();
  cerr << "Processing " << utc_start_str << endl;

  // number of dat processed from the current buffer
  idat_written = 0;
}

void spip::ContainerFileWrite::set_filename (std::string _filename)
{
  filename = _filename;
  fixed_filename = true;
}

void spip::ContainerFileWrite::set_file_length_bytes (int64_t bytes)
{
  desired_bytes = bytes;
  if (verbose)
    cerr << "spip::ContainerFileWrite::set_file_length_bytes bytes=" << bytes << endl;
}

void spip::ContainerFileWrite::set_file_length_seconds (float secs)
{
  desired_secs = secs;
  if (verbose)
    cerr << "spip::ContainerFileWrite::set_file_length_seconds desired_secs=" << secs << endl;
}

void spip::ContainerFileWrite::resize_hdr_buf (unsigned required_size)
{
  if (spip::Container::verbose)
    cerr << "spip::ContainerFileWrite::resize_hdr_buf required_size=" << required_size << endl;
  if (required_size > hdr_bufsz)
  {
    if (spip::Container::verbose)
      cerr << "spip::ContainerFileWrite::resize_hdr_buf resizing from " << hdr_bufsz << endl;
    if (hdr_buf) 
      free (hdr_buf);
    hdr_buf = (char *) malloc (required_size);
    hdr_bufsz = required_size;
  }
}

//! write data stored in the buffer to disk
void spip::ContainerFileWrite::write ()
{
  if (spip::Container::verbose)
    cerr << "spip::ContainerFileWrite::write ndat=" << ndat << endl; 
  for (uint64_t idat=0; idat<ndat; idat++)
  {
    // open the file if required
    if (fd == -1)
    {
      if (spip::Container::verbose)
        cerr << "spip::ContainerFileWrite::write_ndat opening file for idat=" << idat << endl;
      open_file ();
      write_header ();
    }

    // write one data sample at a time to the file
    write_data (idat, 1);

    if (idat_written >= ndat_per_file)
    {
      close_file ();
      idat_written = 0;
    }
  }
}

void spip::ContainerFileWrite::write_header ()
{
  // ensure the local buffer to be large enough for the header
  resize_hdr_buf (hdr_size);
  
  // zero the header buffer
  memset (hdr_buf, '\0', hdr_size);

  if (header.set ("ORDER", "%s", get_order_string(order).c_str()) < 0)
    throw Error (InvalidState, "spip::ContainerFileWrite::write_header", "failed to write ORDER to header");

  // ensure the obs offset is updated
  if (header.set ("OBS_OFFSET", "%lu", obs_offset) < 0)
    throw Error (InvalidState, "spip::ContainerFileWrite::write_header", "failed to write OBS_OFFSET to header");

  file_size = (ndat_per_file * bits_per_sample) / 8;
  if (spip::Container::verbose)
    cerr << "spip::ContainerFileWrite::write_header ndat_per_file=" << ndat_per_file << " file_size=" << file_size << endl;
  if (header.set ("FILE_SIZE", "%lu", file_size) < 0)
    throw Error (InvalidState, "spip::ContainerFileWrite::write_header", "failed to write FILE_SIZE to header");

  // copy the header + null terminating character
  size_t to_copy = strlen (header.raw());
  strncpy (hdr_buf, header.raw(), to_copy);

  // write the full header to the file
  ::write (fd, hdr_buf, hdr_size);
}

uint64_t spip::ContainerFileWrite::write_data (uint64_t start_idat, uint64_t ndat_to_write)
{
  size_t bits_to_write = (ndat_to_write * bits_per_sample);
  if (bits_to_write % 8 != 0)
    throw Error (InvalidState, "spip::ContainerFileWrite::write_data", "can only write integer bytes to file");
  size_t bytes_to_write = bits_to_write / 8;

  uint64_t buffer_offset = (start_idat * bits_per_sample) / 8;

  if ((order != spip::Ordering::TSPF) && (order != spip::Ordering::TFPS) &&
      (order != spip::Ordering::TSPFB))
  {
    throw Error (InvalidState, "spip::ContainerFileWrite::write_data", "unsupported container order");
  }

#ifdef _DEBUG
  cerr << "spip::ContainerFileWrite::write_data start_idat=" << start_idat 
       << " ndat_to_write=" << ndat_to_write << " buffer_offset=" << buffer_offset
       << " bytes_to_write=" << bytes_to_write << endl;
#endif
  uint64_t bytes_written = ::write (fd, buffer + buffer_offset, bytes_to_write);
  if (bytes_written != bytes_to_write)
  {
    throw Error (InvalidState, "spip::ContainerFileWrite::write_data", "failed to write %lu bytes to file, only wrote %lu", bytes_to_write, bytes_written);
  }

  // increment the obs_offset counter
  obs_offset += bytes_written;
  idat_written += ndat_to_write;

  return bytes_written;
}

void spip::ContainerFileWrite::open_file()
{
  if (fd >= 0)
    throw Error(InvalidState, "spip::ContainerFileWrite::open_file", "file descriptor already open");

  int flags = O_WRONLY | O_CREAT | O_TRUNC;
  int perms = S_IRUSR | S_IWUSR | S_IRGRP | S_IROTH;

  std::string utc_start_str = utc_start->get_gmtime();

  if (!fixed_filename)
  {
    char filename_buf[FILENAME_MAX];
    snprintf (filename_buf, FILENAME_MAX, "%s_%016" PRIu64 ".%06lu.dada", utc_start_str.c_str(), obs_offset, file_number);
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

void spip::ContainerFileWrite::close_file()
{
  spip::ContainerFile::close_file();

  // rename temporary file to actual filename
  rename (temporary_filename.c_str(), filename.c_str());
}
