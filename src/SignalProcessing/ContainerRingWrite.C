/***************************************************************************
 *
 *   Copyright (C) 2016 Andrew Jameson
 *   Licensed under the Academic Free License version 2.1
 *
 ***************************************************************************/

#include "spip/ContainerRingWrite.h"

#include <iostream>
#include <cstring>
#include <stdexcept>

using namespace std;

spip::ContainerRingWrite::ContainerRingWrite (spip::DataBlockWrite * _db)
{
  db = _db;
  size = db->get_data_bufsz();
}

spip::ContainerRingWrite::~ContainerRingWrite ()
{
}

//! write output header 
void spip::ContainerRingWrite::process_header ()
{
  // write standard parameters to the ascii header
  write_header();

  // write the ascii header to ring buffer, mark it filled
  db->write_header (header.raw());
}

void spip::ContainerRingWrite::open_block()
{
  unsigned char * tmp = (unsigned char *) db->open_block();
  set_buffer (tmp);
} 

void spip::ContainerRingWrite::close_block()
{
  uint64_t output_bufsz = calculate_buffer_size();
  if (spip::Container::verbose)
    cerr << "spip::ContainerRingWrite::close_block bytes_written=" << output_bufsz << endl; 
  db->close_block (output_bufsz);
  unset_buffer();
}

#ifdef HAVE_CUDA
void spip::ContainerRingWrite::register_buffers()
{
  db->register_cuda();
}
#endif

