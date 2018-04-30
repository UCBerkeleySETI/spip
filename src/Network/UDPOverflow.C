/***************************************************************************
 *
 *   Copyright (C) 2015 Andrew Jameson
 *   Licensed under the Academic Free License version 2.1
 *
 ***************************************************************************/

#include "spip/UDPOverflow.h"

#include <iostream>
#include <cstring>
#include <math.h>

#ifdef __cplusplus
#define __STDC_CONSTANT_MACROS
#ifdef _STDINT_H
#undef _STDINT_H
#endif
#include <stdint.h>
#endif

using namespace std;

spip::UDPOverflow::UDPOverflow()
{
  buffer = NULL;
  bufsz = 0;
  last_byte = 0;
  overflowed_bytes = 0;
}

spip::UDPOverflow::~UDPOverflow()
{
  if (buffer)
    free (buffer);
  buffer = NULL;
}

void spip::UDPOverflow::resize (size_t bytes)
{
  if (bytes > bufsz)
  {
    if (buffer)
      free(buffer);
    buffer = (char *) malloc(bytes);
    bufsz = bytes;
  }
}

void spip::UDPOverflow::reset()
{
  last_byte = 0;
  overflowed_bytes = 0;
}

void spip::UDPOverflow::copied_from (size_t offset, size_t bytes)
{
  last_byte = std::max (offset + bytes, last_byte);
  overflowed_bytes += bytes;
}

void spip::UDPOverflow::copy_from (char * from, size_t offset, size_t bytes)
{
  memcpy (buffer + offset, from, bytes);
  copied_from (offset, bytes);
}

int64_t spip::UDPOverflow::copy_to (char * to)
{
  if (last_byte == 0)
    return 0;

  // copy to the supplied pointer up to the last byte copied in
  memcpy (to, buffer, last_byte);

  // reset the last byte copied
  last_byte = 0;

  int64_t bytes = int64_t(overflowed_bytes);
  bytes = 0;

  // return the total bytes copied from the overflow buffer
  return bytes; 
}
