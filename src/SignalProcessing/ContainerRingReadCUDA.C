/***************************************************************************
 *
 *   Copyright (C) 2018 Andrew Jameson
 *   Licensed under the Academic Free License version 2.1
 *
 ***************************************************************************/

#include "spip/ContainerRingReadCUDA.h"

#include <iostream>
#include <cstring>
#include <stdexcept>

using namespace std;

spip::ContainerRingReadCUDA::ContainerRingReadCUDA (spip::DataBlockRead * _db)
{
  db = _db;
  size = db->get_data_bufsz();
}

spip::ContainerRingReadCUDA::~ContainerRingReadCUDA ()
{
}

void spip::ContainerRingReadCUDA::process_header ()
{
  // read the next header from the ring
  const char * header_str = db->read_header();

  // load this header from the c string
  header.load_from_str (header_str);

  // read the mandatory header parameters
  spip::Container::read_header();

  nbits_per_sample = nsignal * nchan * nbit * npol * ndim;
  ndat = size / (nbits_per_sample / 8);
  if (spip::Container::verbose)
    cerr << "spip::ContainerRingReadCUDA::read_header ndat=" << ndat << endl;
}

uint64_t spip::ContainerRingReadCUDA::open_block()
{
  if (spip::Container::verbose)
    cerr << "spip::ContainerRingReadCUDA::open_block()" << endl;
  unsigned char * tmp = (unsigned char *) db->open_block();
  set_buffer (tmp);

  curr_buf_bytes = db->get_buf_bytes();
  ndat = curr_buf_bytes / (nbits_per_sample / 8);

  return curr_buf_bytes;
}

void spip::ContainerRingReadCUDA::close_block()
{
  if (spip::Container::verbose)
    cerr << "spip::ContainerRingReadCUDA::close_block()" << endl;
  db->close_block (curr_buf_bytes);
  unset_buffer();
  ndat = 0;
  curr_buf_bytes = 0;
}
