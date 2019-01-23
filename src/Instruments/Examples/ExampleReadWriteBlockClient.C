/***************************************************************************
 *
 *   Copyright (C) 2018 Andrew Jameson
 *   Licensed under the Academic Free License version 2.1
 *
 ***************************************************************************/

#include "spip/ExampleReadWriteBlockClient.h"

#include <cstdio>
#include <cstring>
#include <iostream>
#include <stdexcept>

using namespace std;

spip::ExampleReadWriteBlockClient::ExampleReadWriteBlockClient (
    const char * read_key_string,
    const char * write_key_string) :
        spip::ReadWriteBlockClient (read_key_string, write_key_string)
{
}

spip::ExampleReadWriteBlockClient::~ExampleReadWriteBlockClient ()
{
}

int64_t spip::ExampleReadWriteBlockClient::open ()
{
  unsigned nbit;

  // read the headers to determine the offsets for each read pointer
  if (read_header.get ("NBIT", "%u", &nbit) != 1)
    throw invalid_argument ("NBIT did not exist in read_header");

  if (nbit != 16)
    throw invalid_argument ("Only input nbit=16 supported");

  // check the input/output block sizes
  if (read_db->get_data_bufsz() != write_db->get_data_bufsz() * 2)
    throw invalid_argument ("input ring buffer must be double the size of the output ring buffer");

  sample_bytes = nbit / 8;
  scale_factor = 256.0;

  // copy the input header from the appropriate subband
  write_header.clone (read_header);

  // only support output nbit of 8
  unsigned new_nbit = 8;

  write_header.set("NBIT", "%u", new_nbit);

  return 0;
}

int64_t spip::ExampleReadWriteBlockClient::io_block (void * read_buffer,
                                            void * write_buffer,
                                            uint64_t read_bytes)
{
#ifdef _DEBUG
  cerr << "spip::ExampleReadWriteBlockClient::io_block reading " << read_bytes << endl;
#endif

  // input buffer pointers
  int16_t * read = (int16_t *) read_buffer;
  // write buffer pointers
  int8_t * write = (int8_t *) write_buffer;

  // read 1 value at a time.
  uint64_t ndat = read_bytes / sample_bytes;

  //register int16_t converted;
  //register float rescaled;

  // perform the writes from host to device memory
  for (uint64_t idat=0; idat<ndat; idat++)
  {
    const int16_t converted = convert_twos_fast(read[idat]);    
    const float rescaled = float(converted) / scale_factor;
    write[idat] = int8_t(rescaled); 
  }

  int64_t bytes_written = read_bytes / 2;
#ifdef _DEBUG
  cerr << "spip::ExampleReadWriteBlockClient::io_block wrote " << bytes_written << endl;
#endif
  return bytes_written;
}

int64_t spip::ExampleReadWriteBlockClient::close ()
{
  return 0;
}
