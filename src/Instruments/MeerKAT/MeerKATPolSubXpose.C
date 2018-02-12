/***************************************************************************
 *
 *   Copyright (C) 2017 Andrew Jameson
 *   Licensed under the Academic Free License version 2.1
 *
 ***************************************************************************/

#include "spip/MeerKATPolSubXpose.h"

#include <cstdio>
#include <cstring>
#include <iostream>
#include <stdexcept>

using namespace std;

spip::MeerKATPolSubXpose::MeerKATPolSubXpose (const char * read1_key_string,
                                              const char * read2_key_string,
                                              const char * write_key_string,
                                              int _subband) :
  spip::ReadReadWriteBlockClient (read1_key_string, read2_key_string, write_key_string)
{
  subband = _subband;

  // heap size is a constant for MeerKAT's beam-former
  nsamp_per_heap = 256;

  // this operation is hardcoded to 2 sub-bands 
  nsubband = 2;

  in_stride = 0;
  out_stride = 0;
}

spip::MeerKATPolSubXpose::~MeerKATPolSubXpose ()
{
}

int64_t spip::MeerKATPolSubXpose::open ()
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

  // check that npol is indeed 1
  if (npol != 1)
    throw invalid_argument ("NPOL was not equal to 1");

  // copy the input header from the appropriate subband
  // subband0 == pol0,  subband1 == pol1
  write_header.clone (read1_header);


  unsigned new_npol = npol * nsubband;
  unsigned new_nchan = nchan / nsubband;
  double new_bw = bw / subband;
  double new_freq;
  unsigned new_start_channel = start_channel;
  unsigned new_end_channel = end_channel;
  if (subband == 0)
  {
    new_end_channel = end_channel - new_nchan;
    new_freq = freq - new_bw / 2;
  }
  else
  {
    new_start_channel = start_channel + new_nchan;
    new_freq = freq + new_bw / 2;
  }

  write_header.set("NPOL", "%u", new_npol);
  write_header.set("NCHAN", "%u", new_nchan);
  write_header.set("START_CHANNEL", "%u", new_start_channel);
  write_header.set("END_CHANNEL", "%u", new_end_channel);
  write_header.set("BW", "%lf", bw);
  write_header.set("FREQ", "%lf", freq);

  unsigned nbytes_per_heap = (nsamp_per_heap * npol * nchan * ndim * nbit) / 8;
  if (verbose)
    cerr << "spip::MeerKATPolSubXpose::open nbytes_per_heap=" << nbytes_per_heap  << endl;

  // stride between input reads 
  in_stride = nbytes_per_heap;
  in_length = nbytes_per_heap / nsubband;

  // stride between output writes
  out_stride = nbytes_per_heap * new_npol / nsubband;
  out_length = out_stride;

  // offset on first input read
  in1_read_offset = in_length * subband;
  in2_read_offset = in_length * subband;

  // offset on first input write
  in1_write_offset = 0;
  in2_write_offset = in_length;

  if (verbose)
  {
    cerr << "spip::MeerKATPolSubXpose::open in_stride=" << in_stride << " in_length=" << in_length << endl;
    cerr << "spip::MeerKATPolSubXpose::open out_stride=" << out_stride << " out_length=" << out_length << endl;
  }

  return 0;
}

int64_t spip::MeerKATPolSubXpose::io_block (void * read1_buffer, 
                                            void * read2_buffer, 
                                            void * write_buffer, 
                                            uint64_t write_bytes)
{
  cerr << "spip::MeerKATPolSubXpose::io_block" << endl;

  uint64_t nblocks = write_bytes / out_length;

  // input buffer pointers
  char * read1 = (char *) read1_buffer;
  char * read2 = (char *) read2_buffer;

  // write buffer pointers
  char * write1 = (char *) write_buffer;
  char * write2 = (char *) write_buffer;

  // increment read pointers by their associated offsets
  read1 += in1_read_offset;
  read2 += in2_read_offset;

  write1 += in1_write_offset;
  write2 += in2_write_offset;

  // perform the writes from host to device memory
  for (uint64_t iblock=0; iblock<nblocks; iblock++)
  {
    memcpy (write1, read1, in_length);
    memcpy (write2, read2, in_length);

    // increment the input pointers 
    read1 += in_stride;
    read2 += in_stride;

    // increment the output pointers
    write1 += out_stride;
    write2 += out_stride;
  }

  return write_bytes;
}

int64_t spip::MeerKATPolSubXpose::close ()
{
  return 0;
}
