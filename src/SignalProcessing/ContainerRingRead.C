/***************************************************************************
 *
 *   Copyright (C) 2016 Andrew Jameson
 *   Licensed under the Academic Free License version 2.1
 *
 ***************************************************************************/

#include "spip/ContainerRingRead.h"
#include "spip/AsciiHeader.h"

#include <iostream>
#include <cstring>
#include <stdexcept>

using namespace std;

spip::ContainerRingRead::ContainerRingRead (spip::DataBlockRead * _db)
{
  db = _db;
  size = db->get_data_bufsz();
}

spip::ContainerRingRead::~ContainerRingRead ()
{
}

void spip::ContainerRingRead::process_header ()
{
  // read the next header from the ring
  const char * header_str = db->read_header();

  // load this header from the c string
  if (spip::Container::verbose)
    cerr << "spip::ContainerRingRead::process_header header.load_from_str()" << endl;
  header.load_from_str (header_str);

  // read the mandatory header parameters
  if (spip::Container::verbose)
    cerr << "spip::ContainerRingRead::process_header read_header" << endl;
  spip::Container::read_header();

  // ensure ndat is set based on header size and parameters
  if (spip::Container::verbose)
    cerr << "spip::ContainerRingRead::process_header size=" << size << " nbytes/sample=" << (bits_per_sample / 8) << endl;
  ndat = size / (bits_per_sample / 8);
  if (spip::Container::verbose)
    cerr << "spip::ContainerRingRead::read_header ndat=" << ndat << endl;

  // TODO should this be shifted to read_header()? 
  calculate_strides ();
}

uint64_t spip::ContainerRingRead::open_block()
{
  unsigned char * tmp = (unsigned char *) db->open_block();
  set_buffer (tmp);

  curr_buf_bytes = db->get_buf_bytes();
  ndat = curr_buf_bytes / (bits_per_sample / 8);

  return curr_buf_bytes;
}

void spip::ContainerRingRead::close_block()
{
  db->close_block (curr_buf_bytes);
  unset_buffer(); 
  ndat = 0;
  curr_buf_bytes = 0;
}

#ifdef HAVE_CUDA
void spip::ContainerRingRead::register_buffers()
{
  db->register_cuda();
}
#endif

