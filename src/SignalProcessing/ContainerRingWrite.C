/***************************************************************************
 *
 *   Copyright (C) 2016 Andrew Jameson
 *   Licensed under the Academic Free License version 2.1
 *
 ***************************************************************************/

#include "spip/Error.h"
#include "spip/ContainerRingWrite.h"

#include <iostream>
#include <cstring>
#include <stdexcept>

using namespace std;

spip::ContainerRingWrite::ContainerRingWrite (spip::DataBlockWrite * _db)
{
  db = _db;
  size = db->get_data_bufsz();
#ifdef HAVE_CUDA
  if (db->get_device() != -1)
    throw Error (InvalidState, "spip::ContainerRingWrite::ContainerRingWrite",
                 "Cannot operate on GPU ring buffer");
#endif
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

uint64_t spip::ContainerRingWrite::open_block()
{
  unsigned char * tmp = (unsigned char *) db->open_block();
  if (spip::Container::verbose)
    cerr << "spip::ContainerRingWrite::open_block buffer_ptr=" << (void *) tmp << endl;
  set_buffer (tmp);
  return size;
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

