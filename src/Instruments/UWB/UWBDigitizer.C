/***************************************************************************
 *
 *   Copyright (C) 2017 Andrew Jameson
 *   Licensed under the Academic Free License version 2.1
 *
 ***************************************************************************/

#include "spip/UWBDigitizer.h"

#include <cstdio>
#include <cstring>
#include <iostream>
#include <stdexcept>

using namespace std;

spip::UWBDigitizer::UWBDigitizer (const char * read_key_string,
                                  const char * write_key_string) :
  spip::ReadWriteBlockClient (read_key_string, write_key_string)
{
  nbit_out = 8;
  nsamp_per_block = 1024;
  int16_t scale_16 = 256;
}

spip::UWBDigitizer::~UWBDigitizer ()
{
}

int64_t spip::UWBDigitizer::open ()
{
  unsigned npol, nchan, nbit, ndim, start_channel, end_channel;
  double bw, freq;

  // read the headers to determine the offsets for each read pointer
  if (read1_header.get ("NPOL", "%u", &npol) != 1)
    throw invalid_argument ("NPOL did not exist in read1_header"); 
  if (read1_header.get ("NCHAN", "%u", &nchan) != 1)
    throw invalid_argument ("NCHAN did not exist in read1_header"); 
  if (read1_header.get ("NBIT", "%u", &nbit) != 1)
    throw invalid_argument ("NBIT did not exist in read1_header"); 
  if (read1_header.get ("NDIM", "%u", &ndim) != 1)
    throw invalid_argument ("NDIM did not exist in read1_header"); 
  if (read1_header.get ("BW", "%lf", &bw) != 1)
    throw invalid_argument ("BW did not exist in read1_header"); 
  if (read1_header.get ("FREQ", "%lf", &freq) != 1)
    throw invalid_argument ("FREQ did not exist in read1_header"); 
  if (read1_header.get ("START_CHANNEL", "%u", &start_channel) != 1)
    throw invalid_argument ("START_CHANNEL did not exist in read1_header"); 
  if (read1_header.get ("END_CHANNEL", "%u", &start_channel) != 1)
    throw invalid_argument ("END_CHANNEL did not exist in read1_header"); 

  // check that npol is indeed 2
  if (npol != 2)
    throw invalid_argument ("NPOL was not equal to 2");

  // copy the header from the input
  write_header.clone (read_header);

  write_header.set("NBIT", "%u", nbit_out);

  return 0;
}

int64_t spip::UWBDigitizer::io_block (void * read_buffer, 
                                      void * write_buffer, 
                                      uint64_t read_bytes)
{
  if (nbit_out == 16)
    return io_block_16 (read_buffer, write_buffer, write_bytes);    
  else if (nbit_out == 8)
    return io_block_8 (read_buffer, write_buffer, write_bytes);    
  else
    throw (invalid_argument ("output nbits must be 8 or 16");

  return -1;
}

int64_t spip::UWBDigitizer::io_block_16 (void * read_buffer,
                                         void * write_buffer,
                                         uint64_t read_bytes)
{
  memcpy (write_buffer,, read_buffer, read_bytes);
  return int64_t(read_bytes);
}

int64_t spip::UWBDigitizer::io_block_8 (void * read_buffer,
                                        void * write_buffer,
                                        uint64_t read_bytes)
{
  uint64_t nval = (read_bytes * 8) / nbit;

  // input buffer pointer
  int16_t * read = (int16_t *) read_buffer;
  int8_t * write = (int8_t *) write_buffer;

  // use the first block of data to determine the scale
  for (uint64_t ival = 0; ival<nval; ival++)
  {
    write[ival] = read[ival] / scale_16;
  }
  return int64_t(read_bytes);
}

int64_t spip::UWBDigitizer::measure_variance (void * read_buffer,
                                              uint64_t write_bytes)
{
  // TODO
}


int64_t spip::UWBDigitizer::close ()
{
  return 0;
}
