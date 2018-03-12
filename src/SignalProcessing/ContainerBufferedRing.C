/***************************************************************************
 *
 *   Copyright (C) 2016 Andrew Jameson
 *   Licensed under the Academic Free License version 2.1
 *
 ***************************************************************************/

#include "spip/ContainerBufferedRing.h"
#include "spip/AsciiHeader.h"
#include "spip/Error.h"

#include <iostream>
#include <cstring>
#include <stdexcept>

using namespace std;

spip::ContainerBufferedRing::ContainerBufferedRing ()
{
  if (spip::Container::verbose)
    cerr << "spip::ContainerBufferedRing ctor" << endl;
}

spip::ContainerBufferedRing::~ContainerBufferedRing ()
{
  if (spip::Container::verbose)
    cerr << "spip::ContainerBufferedRing dtor" << endl;

  if (buffer)
    free (buffer);
  buffer = NULL;
}

void spip::ContainerBufferedRing::resize ()
{
  if (spip::Container::verbose)
    cerr << "spip::ContainerBufferedRing::resize ndat=" << ndat << " nchan=" << nchan 
         << " nsignal=" << nsignal << " ndim=" << ndim << " npol=" << npol << " nbit=" << nbit << endl;

  uint64_t required_size = calculate_buffer_size ();
  if (spip::Container::verbose)
    cerr << "spip::ContainerBufferedRing::resize size=" << size << " required_size=" << required_size << endl;
  if (required_size > size)
  {
    if (buffer)
      free (buffer);
    buffer = (unsigned char *) malloc (required_size);
    if (!buffer)
      throw Error(InvalidState, "spip::ContainerBufferedRing::resize", "malloc failed");
    size = required_size;
  }
}

void spip::ContainerBufferedRing::zero ()
{
  bzero (buffer, size);
}
