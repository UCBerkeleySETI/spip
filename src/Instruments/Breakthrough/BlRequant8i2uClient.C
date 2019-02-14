/***************************************************************************
 *
 *   Copyright (C) 2018 Andrew Jameson
 *   Licensed under the Academic Free License version 2.1
 *
 ***************************************************************************/

#include "spip/BlRequant8i2uClient.h"

#include <cstdio>
#include <cstring>
#include <iostream>
#include <stdexcept>
#include "math.h" 

using namespace std;


spip::BlRequant8i2uClient::BlRequant8i2uClient (
    const char * read_key_string,
    const char * write_key_string) :
        spip::ReadWriteBlockClient (read_key_string, write_key_string)
{
}

spip::BlRequant8i2uClient::~BlRequant8i2uClient ()
{
}

int64_t spip::BlRequant8i2uClient::open ()
{
  unsigned nbit;

  // read the headers to determine the offsets for each read pointer
  if (read_header.get ("NBIT", "%u", &nbit) != 1)
    throw invalid_argument ("NBIT did not exist in read_header");

  if (nbit != 8)
    throw invalid_argument ("Only input nbit=8 supported");

  // check the input/output block sizes
  if (read_db->get_data_bufsz() != write_db->get_data_bufsz() * 4)
    throw invalid_argument ("input ring buffer must be 4x the size of the output ring buffer");

  sample_bytes = 1;
  scale_factor = 256.0;

  // copy the input header from the appropriate subband
  write_header.clone (read_header);

  // only support output nbit of 8
  unsigned new_nbit = 2;

  write_header.set("NBIT", "%u", new_nbit);

  return 0;
}

int64_t spip::BlRequant8i2uClient::io_block (void * read_buffer,
                                            void * write_buffer,
                                            uint64_t read_bytes)
{
#ifdef _DEBUG
  cerr << "spip::BlRequant8i2uClient::io_block reading " << read_bytes << endl;
#endif

  // input buffer pointers
  int8_t * read = (int8_t *) read_buffer;
  // write buffer pointers
  int8_t * write = (int8_t *) write_buffer;

  // read 1 value at a time.
  uint64_t ndat = read_bytes;

  // Compute STDEV of REAL and IMAG components
  double sum_re = 0.0;
  double sq_sum_re = 0.0;
  double sum_im = 0.0;
  double sq_sum_im = 0.0;
  for(size_t idx = 0; idx < ndat/2; idx++){
      sum_re += read[2*idx];
      sq_sum_re += read[2*idx] * read[2*idx];
      sum_im += read[2*idx+1];
      sq_sum_im += read[2*idx+1] * read[2*idx+1];
  }
  double mean_re = sum_re / ndat;
  double stdev_re = sqrt(sq_sum_re / ndat - mean_re * mean_re);
  double mean_im = sum_im / ndat;
  double stdev_im = sqrt(sq_sum_im / ndat - mean_im * mean_im);

  // Do 2-bit conversion
  for(size_t idx = 0; idx < ndat / 4; idx++) {

      // We are going to add all 2-bits together into one 8-bit number
      // So break out each into indexes
      size_t idxr = 4*idx;
      size_t idxi = 4*idx + 1;
      size_t idxr2 = 4*idx + 2;
      size_t idxi2 = 4*idx + 3;
      //std::cout << read[idxr] << " " << read[idxi] << " ";

      // Real part
      if(read[idxr] <  -0.98159883*stdev_re) {
          write[idx] += 0 * 64;
      } else if(read[idxr] < 0){
          write[idx] += 1 * 64;
      } else if(read[idxr] < 0.98159883*stdev_re) {
          write[idx] += 2 * 64;
      } else {
          write[idx] += 3 * 64;
      }

      if(read[idxr2] <  -0.98159883*stdev_re) {
          write[idx] += 0 * 4;
      } else if(read[idxr2] < 0){
          write[idx] += 1 * 4;
      } else if(read[idxr2] < 0.98159883*stdev_re) {
          write[idx] += 2 * 4;
      } else {
          write[idx] += 3 * 4;
      }

      // Imag part
      if(read[idxi] <  -0.98159883*stdev_im) {
          write[idx] += 0 * 16;
      } else if(read[idxi] < 0) {
          write[idx] += 1 * 16;
      } else if(read[idxi] < 0.98159883*stdev_im) {
          write[idx] += 2 * 16;
      } else {
          write[idx] += 3 * 16;
      }

      if(read[idxi2] <  -0.98159883*stdev_im) {
          write[idx] += 0;
      } else if(read[idxi2] < 0) {
          write[idx] += 1;
      } else if(read[idxi2] < 0.98159883*stdev_im) {
          write[idx] += 2;
      } else {
          write[idx] += 3;
      }

  }

  int64_t bytes_written = read_bytes / 4;
#ifdef _DEBUG
  cerr << "spip::BlRequant8i2uClient::io_block wrote " << bytes_written << endl;
#endif
  return bytes_written;
}

int64_t spip::BlRequant8i2uClient::close ()
{
  return 0;
}
