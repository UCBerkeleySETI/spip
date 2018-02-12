/***************************************************************************
 *
 *   Copyright (C) 2016 Andrew Jameson
 *   Licensed under the Academic Free License version 2.1
 *
 ***************************************************************************/

#include "spip/ContainerRing.h"
#include "spip/AsciiHeader.h"

#include <iostream>
#include <cstring>
#include <stdexcept>

using namespace std;

spip::ContainerRing::ContainerRing ()
{
  buffer_valid = false;
}

spip::ContainerRing::~ContainerRing ()
{
}

void spip::ContainerRing::resize ()
{
  if (spip::Container::verbose)
    cerr << "spip::ContainerRing::resize ndat=" << ndat << " nchan=" << nchan << " nsignal=" << nsignal << " ndim=" << ndim << " npol=" << npol << " nbit=" << nbit << endl;

  uint64_t required_size = calculate_buffer_size ();
  if (spip::Container::verbose)
    cerr << "spip::ContainerRing::resize size=" << size << " required_size=" << required_size << endl;
  if (required_size > size)
  {
    throw runtime_error ("required size for container not equal to ring buffer size");
  }
}

void spip::ContainerRing::zero ()
{
  if (buffer_valid)
    bzero (buffer, size);
  else
    throw runtime_error ("cannot zero an invalid buffer");
}

void spip::ContainerRing::set_buffer (unsigned char * buf)
{
  buffer = buf;
  buffer_valid = true;
}

void spip::ContainerRing::unset_buffer ()
{
  buffer = NULL;
  buffer_valid = false;
}
