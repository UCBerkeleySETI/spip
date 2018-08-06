/***************************************************************************
 *
 *   Copyright (C) 2015 Andrew Jameson
 *   Licensed under the Academic Free License version 2.1
 *
 ***************************************************************************/

#include "spip/UDPFormat.h"

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

inline int32_t convert_offset_binary (int32_t in) { return in^0x8000000; };
inline int16_t convert_offset_binary (int16_t in) { return in^0x8000; };
inline int8_t convert_offset_binary (int8_t in) { return in^0x80; };

spip::UDPFormat::UDPFormat()
{
  // some defaults
  npol = 2;
  ndim = 2;
  nbit = 8;
  packet_header_size = 8;
  packet_data_size   = 1024;

  noise_buffer = 0;
  noise_buffer_size = 1048576;
  noise_buffer_alignment = 1;

  // default twos complement and little endian
  twos_complement = false;
  little_endian = true;

  prepared = false;
  configured = false;
  self_start = false;
}

spip::UDPFormat::~UDPFormat()
{
}

double spip::UDPFormat::rand_normal (double mean, double stddev)
{
  n2 = 0.0;
  n2_cached = 0;

  if (!n2_cached) 
  {
    // Choose a point x,y in the unit circle uniformaly at random
    double x, y, r;
    do {
      // scale two random integers to doubles between -1 and 1
      x = 2.0*rand()/RAND_MAX - 1;
      y = 2.0*rand()/RAND_MAX - 1;

      r = x*x + y*y;
    } while (r == 0.0 || r > 1.0);

    {
      // Apply Box-Muller transform on x,y
      double d = sqrt(-2.0*log(r)/r);
      double n1 = x*d;
      n2 = y*d;

      // Scale and translate to get desired mean and standard deviation
      double result = n1*stddev + mean;

      n2_cached = 1;
      return result;
    }
  } else {
    n2_cached = 0;
    return n2*stddev + mean;
  }
}

void spip::UDPFormat::set_noise_buffer_size (unsigned nbytes)
{
  noise_buffer_size = nbytes;

  if (noise_buffer)
  {
    free (noise_buffer);
    noise_buffer = (char *) malloc (noise_buffer_size);
  }
}


void spip::UDPFormat::generate_noise_buffer (int nbits)
{
  if (!noise_buffer)
    noise_buffer = (char *) malloc (noise_buffer_size);

  // generate noise with mean of 0 and stddev of 0.078125 (10 / 128)
  noise_buffer_alignment = nbits / 8;
  const double mean = 0;
  //const double stddev = (10.0 / 128.0) * pow(2, (nbits-1));
  const double stddev = (20.0 / 128.0) * pow(2, (7));
  const unsigned size = (noise_buffer_size * 8) / nbits;
  int8_t * buffer8 = (int8_t *) noise_buffer;
  int16_t * buffer16 = (int16_t *) noise_buffer;
  int32_t * buffer32 = (int32_t *) noise_buffer;

  // seed the random number generator
  srand(time(0));

  for (unsigned i=0; i < size; i++)
  {
    double val = rint(rand_normal (mean, stddev)); 

    if (nbits == 8)
    {
      if (twos_complement)
        buffer8[i] = (int8_t) val;
      else
        buffer8[i] = convert_offset_binary((int8_t) val);
    }
    else if (nbits == 16)
    {
      if (twos_complement)
        buffer16[i] = (int16_t) val;
      else
        buffer16[i] = convert_offset_binary ((int16_t) val);
    }
    else if (nbits == 32)
    {
      if (twos_complement)
        buffer32[i] = (int32_t) val;
      else
        buffer32[i] = convert_offset_binary((int32_t) val);
    }
    else
      return;
  }
}

void spip::UDPFormat::fill_noise (char * buf, size_t nbytes)
{
  // choose a random starting point in the noise buffer
  int start_byte = (int) floor( ((double) rand() * noise_buffer_size) / RAND_MAX);
  if (start_byte + nbytes > noise_buffer_size)
    start_byte -= nbytes;

  // ensure correct alignment (for multi byte sampling)
  int remainder = start_byte % noise_buffer_alignment;
  start_byte -= remainder;

  // copy from the buffer
  memcpy (buf, noise_buffer + start_byte, nbytes);
}
