/***************************************************************************
 *
 *   Copyright (C) 2018 Andrew Jameson
 *   Licensed under the Academic Free License version 2.1
 *
 ***************************************************************************/

#include "spip/ContainerFileRead.h"
#include "spip/AsciiHeader.h"
#include "spip/Error.h"

#include <iostream>
#include <cstring>
#include <stdexcept>

#include <unistd.h>
#include <fcntl.h>

using namespace std;

spip::ContainerFileRead::ContainerFileRead (std::string _filename)
{
  filename = _filename;
}

spip::ContainerFileRead::~ContainerFileRead ()
{
}

void spip::ContainerFileRead::process_header()
{
  spip::Container::read_header();

  // determine the number of data points
  set_ndat ((data_size * 8) / bits_per_sample);

  // ensure enough space exists
  resize();  
}

//! read data into the container
uint64_t spip::ContainerFileRead::read_data()
{
  if (spip::Container::verbose)
    cerr << "spip::ContainerFileRead::read_data reading " << file_size 
         << " bytes into " << (void *) buffer << endl;

  size_t total_bytes_read = 0;
  while (total_bytes_read < file_size)
  {
    size_t bytes_to_read = file_size - total_bytes_read;

    // read from the FD into the containers buffer the file size listed in the header
    size_t bytes_read = ::read (fd, buffer, file_size);
    if (bytes_read < 0)
      throw Error (InvalidState, "spip::ContainerFileRead::read_data", "read failed to read data from file");
    total_bytes_read += bytes_read;

  }
  if (total_bytes_read != file_size)
    throw Error (InvalidState, "spip::ContainerFileRead::read_data", "advertised data size was %ld, but only %ld bytes were read", file_size, total_bytes_read);

  return uint64_t(total_bytes_read);
}

void spip::ContainerFileRead::open_file()
{
  if (fd >= 0)
    throw Error(InvalidState, "spip::ContainerFileRead::open_file", "file descriptor already open");

  int flags = O_RDONLY;
  int perms = S_IRUSR | S_IWUSR | S_IRGRP | S_IROTH;

  // the filesize of the whole file
  file_size_total = spip::AsciiHeader::filesize (filename.c_str());
  if (spip::Container::verbose)
    cerr << "spip::ContainerFileRead::open_file file_size_total=" << file_size_total << endl;

  // read the header first
  header.load_from_file (filename.c_str());

  if (header.get("HDR_SIZE", "%u", &hdr_size) != 1)
    throw Error (InvalidState, "spip::ContainerFileRead::open_file", "header was missing HDR_SIZE");

  if (spip::Container::verbose)
    cerr << "spip::ContainerFileRead::open_file hdr_size=" << hdr_size << endl;

  data_size = file_size_total - hdr_size;

  if (spip::Container::verbose)
    cerr << "spip::ContainerFileRead::open_file data_size=" << data_size << endl;

  fd = open (filename.c_str(), flags, perms);
  off_t offset = lseek (fd, hdr_size, SEEK_SET);
  if (offset != hdr_size)
    throw Error (InvalidState, "spip::ContainerFileRead::open_file", "failed to seek past header offset=%d", offset);
}
