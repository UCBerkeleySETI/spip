/***************************************************************************
 *
 *   Copyright (C) 2016 Andrew Jameson
 *   Licensed under the Academic Free License version 2.1
 *
 ***************************************************************************/

#include "spip/ContainerBufferedRingWrite.h"

#include <iostream>
#include <cstring>
#include <stdexcept>

using namespace std;

spip::ContainerBufferedRingWrite::ContainerBufferedRingWrite (spip::DataBlockWrite * _db)
{
  db = _db;
}

spip::ContainerBufferedRingWrite::~ContainerBufferedRingWrite ()
{
}

//! write output header 
void spip::ContainerBufferedRingWrite::process_header ()
{
  // write standard parameters to the ascii header
  write_header();

  // write the ascii header to ring buffer, mark it filled
  db->write_header (header.raw());
}

//! write the contents of buffer to the ring
void spip::ContainerBufferedRingWrite::write_buffer ()
{
  // ensure we write only the size of the buffer
  size_t to_write = size_t(calculate_buffer_size ());
  if (spip::Container::verbose)
    cerr << "spip::ContainerBufferedRingWrite::write_buffer to_write=" << to_write << endl;
  if (to_write > 0)
  {
    db->write_data (buffer, to_write);
  }
} 

#ifdef HAVE_CUDA
void spip::ContainerBufferedRingWrite::register_buffers()
{
  db->register_cuda();
}
#endif

