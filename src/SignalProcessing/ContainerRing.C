/***************************************************************************
 *
 *   Copyright (C) 2016 Andrew Jameson
 *   Licensed under the Academic Free License version 2.1
 *
 ***************************************************************************/

#include "spip/Error.h"
#include "spip/ContainerRing.h"
#include "spip/AsciiHeader.h"

#include <iostream>
#include <cstring>

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
    throw Error (InvalidState, "spip::ContainerRing::resize",
                 "required size for container [%lu] not equal to data block buffer size [%lu]\n",
                 required_size, size);
  }
}

void spip::ContainerRing::zero ()
{
  if (buffer_valid)
    bzero (buffer, size);
  else
    throw Error (InvalidState, "spip::ContainerRing::zero", "cannot zero an invalid buffer");
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

