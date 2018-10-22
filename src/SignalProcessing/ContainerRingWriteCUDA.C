/***************************************************************************
 *
 *   Copyright (C) 2018 Andrew Jameson
 *   Licensed under the Academic Free License version 2.1
 *
 ***************************************************************************/

#include "spip/ContainerRingWriteCUDA.h"

#include <iostream>
#include <cstring>
#include <stdexcept>

using namespace std;

spip::ContainerRingWriteCUDA::ContainerRingWriteCUDA (spip::DataBlockWrite * _db)
{
  db = _db;
  size = db->get_data_bufsz();
}

spip::ContainerRingWriteCUDA::~ContainerRingWriteCUDA ()
{
}

void spip::ContainerRingWriteCUDA::zero ()
{
  if (is_valid())
    cudaMemsetAsync (buffer, 0, size);
  else
    throw runtime_error ("cannot zero an invalid buffer");
}


//! write output header 
void spip::ContainerRingWriteCUDA::process_header ()
{
  // write standard parameters to the ascii header
  write_header();

  // write the ascii header to ring buffer, mark it filled
  db->write_header (header.raw());
}

uint64_t spip::ContainerRingWriteCUDA::open_block()
{
  unsigned char * tmp = (unsigned char *) db->open_block();
  set_buffer (tmp);
  return size; 
} 

void spip::ContainerRingWriteCUDA::close_block()
{
  uint64_t output_bufsz = calculate_buffer_size();
  if (spip::Container::verbose)
    cerr << "spip::ContainerRingWriteCUDA::close_block bytes_written=" << output_bufsz << endl; 
  db->close_block (output_bufsz);
  unset_buffer();
}
