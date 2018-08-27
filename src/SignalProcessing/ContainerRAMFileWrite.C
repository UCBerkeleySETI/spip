/***************************************************************************
 *
 *   Copyright (C) 2018 Andrew Jameson
 *   Licensed under the Academic Free License version 2.1
 *
 ***************************************************************************/

#include "spip/ContainerRAMFileWrite.h"
#include "spip/AsciiHeader.h"
#include "spip/Error.h"

#include <iostream>
#include <cstring>

#include <unistd.h>
#include <fcntl.h>

using namespace std;

spip::ContainerRAMFileWrite::ContainerRAMFileWrite (std::string _dir) : FileWrite (_dir)
{
}

spip::ContainerRAMFileWrite::~ContainerRAMFileWrite ()
{
}

void spip::ContainerRAMFileWrite::process_header ()
{
  spip::FileWrite::configure (dynamic_cast<Container *>(this));
}

//! write data stored in the buffer to disk
void spip::ContainerRAMFileWrite::write (uint64_t ndat)
{
  if (spip::File::verbose)
    cerr << "spip::FileWrite::write ndat=" << ndat << endl;

  for (uint64_t idat=0; idat<ndat; idat++)
  {
    // open the file if required
    if (fd == -1)
    {
      std::string utc_start_str = utc_start->get_gmtime();
      if (spip::File::verbose)
        cerr << "spip::FileWrite::write_ndat open_file (" << utc_start 
             << ", " << obs_offset << ") for idat=" << idat << endl;
      spip::FileWrite::open_file (utc_start_str.c_str(), obs_offset);

      // write out the current header
      write_header ();
    }

    // write one data sample at a time to the file
    // TODO this may not be efficient! consider paramaterizing ndat_per_write
    write_data (idat, 1);
    
    if (idat_written >= ndat_per_file)
    {
      close_file (); 
      idat_written = 0;
    }
  }
}

void spip::ContainerRAMFileWrite::write_header ()
{
  // update parameters that may vary from file to file
  if (header.set ("OBS_OFFSET", "%lu", obs_offset) < 0)
    throw Error (InvalidState, "spip::FileWrite::write_header",
                 "failed to write OBS_OFFSET to header");

  file_size = (ndat_per_file * bits_per_sample) / 8;
  if (spip::File::verbose)
    cerr << "spip::FileWrite::write_header ndat_per_file=" << ndat_per_file 
         << " file_size=" << file_size << endl;
  if (header.set ("FILE_SIZE", "%lu", file_size) < 0)
    throw Error (InvalidState, "spip::FileWrite::write_header", 
                 "failed to write FILE_SIZE to header");

  // write the header to file
  spip::FileWrite::write_header (header.raw(), hdr_size);
}


uint64_t spip::ContainerRAMFileWrite::write_data (uint64_t start_idat, uint64_t ndat_to_write)
{
  size_t bits_to_write = (ndat_to_write * bits_per_sample);
  if (bits_to_write % 8 != 0)
    throw Error (InvalidState, "spip::ContainerRAMFileWrite::write_data", 
                 "can only write integer bytes to file");
  size_t bytes_to_write = bits_to_write / 8;

  uint64_t buffer_offset = (start_idat * bits_per_sample) / 8;

  if ((order != spip::Ordering::TSPF) && (order != spip::Ordering::TFPS) &&
      (order != spip::Ordering::TSPFB))
  {
    throw Error (InvalidState, "spip::ContainerRAMFileWrite::write_data", 
                 "unsupported container order");
  }

#ifdef _DEBUG
  cerr << "spip::ContainerRAMFileWrite::write_data start_idat=" << start_idat 
       << " ndat_to_write=" << ndat_to_write << " buffer_offset=" << buffer_offset
       << " bytes_to_write=" << bytes_to_write << endl;
#endif

  uint64_t bytes_written = write_data_bytes (fd, buffer + buffer_offset, bytes_to_write, ndat_to_write);
  if (bytes_written != bytes_to_write)
  {
    throw Error (InvalidState, "spip::ContainerRAMFileWrite::write_data", 
                 "failed to write %lu bytes to file, only wrote %lu", 
                 bytes_to_write, bytes_written);
  }

  // increment the obs_offset counter
  obs_offset += bytes_written;
  idat_written += ndat_to_write;

  return bytes_written;
}
