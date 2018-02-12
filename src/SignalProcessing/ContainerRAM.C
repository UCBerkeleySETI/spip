/***************************************************************************
 *
 *   Copyright (C) 2016 Andrew Jameson
 *   Licensed under the Academic Free License version 2.1
 *
 ***************************************************************************/

#include "spip/ContainerRAM.h"
#include "spip/Error.h"

#include <cstring>
#include <iostream>


using namespace std;

spip::ContainerRAM::ContainerRAM ()
{
}

spip::ContainerRAM::~ContainerRAM ()
{
  // deallocate any buffer
  if (buffer)
    free (buffer);
  buffer = NULL;
}

void spip::ContainerRAM::resize ()
{
  uint64_t required_size = calculate_buffer_size ();
  if (required_size > size)
  {
#ifdef _DEBUG
    cerr << "spip::ContainerRAM::resize resizing from " << size << " to " << required_size << " bytes" << endl;
#endif
    if (buffer)
      free (buffer);
    buffer = (unsigned char *) malloc (required_size);
    if (!buffer)
      throw Error(InvalidState, "spip::ContainerRAM::resize", "malloc failed");

    size = required_size;
  }
}

void spip::ContainerRAM::zero ()
{
  bzero (buffer, size);
}
