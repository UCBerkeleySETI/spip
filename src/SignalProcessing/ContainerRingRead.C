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
  header.load_from_str (header_str);

  if (header.get ("NANT", "%u", &nsignal) != 1)
    throw invalid_argument ("NANT did not exist in header");

  if (header.get ("NCHAN", "%u", &nchan) != 1)
    throw invalid_argument ("NCHAN did not exist in header");

  if (header.get ("NBIT", "%u", &nbit) != 1)
    throw invalid_argument ("NBIT did not exist in header");

  if (header.get ("NPOL", "%u", &npol) != 1)
    throw invalid_argument ("NPOL did not exist in header");

  if (header.get ("NDIM", "%u", &ndim) != 1)
    throw invalid_argument ("NDIM did not exist in header");

  char order[5];
  if (header.get ("ORDER", "%s", order) != 1)
    throw invalid_argument ("ORDER did not exist in header");
  if (strcmp (order, "TSPF") == 0)
    set_order (spip::Ordering::TSPF);
  else if (strcmp (order, "SFPT") == 0)
    set_order (spip::Ordering::SFPT);
  else
    throw invalid_argument ("unsupported input order");


  nbits_per_sample = nsignal * nchan * nbit * npol * ndim;
  ndat = size / (nbits_per_sample / 8);
  if (spip::Container::verbose)
    cerr << "spip::ContainerRingRead::read_header ndat=" << ndat << endl;
}

uint64_t spip::ContainerRingRead::open_block()
{
  unsigned char * tmp = (unsigned char *) db->open_block();
  set_buffer (tmp);

  curr_buf_bytes = db->get_buf_bytes();
  ndat = curr_buf_bytes / (nbits_per_sample / 8);

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

