/***************************************************************************
 *
 *   Copyright (C) 2018 Andrew Jameson
 *   Licensed under the Academic Free License version 2.1
 *
 ***************************************************************************/

// #define _DEBUG

#include "spip/UDPFormatUWB.h"
#include "spip/Error.h"

#include <cstdio>
#include <cstdlib>
#include <iostream>
#include <stdexcept>

using namespace std;

spip::UDPFormatUWB::UDPFormatUWB (int pps) : spip::UDPFormatVDIF (pps)
{
}

spip::UDPFormatUWB::~UDPFormatUWB()
{
}

void spip::UDPFormatUWB::configure (const spip::AsciiHeader& config, const char* suffix)
{
#ifdef _DEBUG
  cerr << "spip::UDPFormatUWB::configure()" << endl;
#endif

  spip::UDPFormatVDIF::configure (config, suffix);

  // ensure that the header_npol is 2 or 3
  if (header_npol != 2 || header_npol != 3)
    throw Error (InvalidState, "spip::UDPFormatUWB::configure", "npol in header was not 2 or 3");
}

uint64_t spip::UDPFormatUWB::get_resolution ()
{
  return header_npol * packet_data_size;
}

// TODO check whether npol or header_npol
uint64_t spip::UDPFormatUWB::get_samples_for_bytes (uint64_t nbytes)
{
  cerr << "spip::UDPFormatUWB::get_samples_for_bytes npol=" << npol 
       << " ndim=" << ndim << " nchan=" << nchan << endl;
  uint64_t nsamps = nbytes / (npol * ndim * nchan);

  return nsamps;
}

inline int64_t spip::UDPFormatUWB::decode_packet (char * buf, unsigned * pkt_size)
{
  // header is stored in the front of the packet
  vdif_header * header_ptr = (vdif_header *) buf;

  // this UDP format assumes 1 thread, for either pol 0 or 1
  int thread_id = getVDIFThreadID (header_ptr);

  if ((thread_id < 0) || (thread_id > 2))
    throw Error (InvalidState, "spip::UDPFormatUWB::decode_packet", "VDIF thread ID must be 0, 1 or 2");

  // configure the stream on the first received packet, to find the start_second
  if (!configured_stream)
  {
    configure_stream (buf);
  }

  *pkt_size = packet_data_size;

  payload = buf + packet_header_size;

  // extract key parameters from the header
  const int offset_second = getVDIFFrameEpochSecOffset (header_ptr) - start_second;
  const int frame_number  = getVDIFFrameNumber (header_ptr);

  // calculate the byte offset for this frame within the data stream
  int64_t byte_offset = (bytes_per_second * offset_second) + (frame_size * frame_number) + (thread_id * packet_data_size);

  return (int64_t) byte_offset;
}

// generate the next packet in the sequence
inline void spip::UDPFormatUWB::gen_packet (char * buf, size_t bufsz)
{
  // packets are packed in TF order 

  // write the new header
  encode_header (buf);

  // increment our "channel number"
  header.threadid++;

  if (header.threadid > end_channel)
  {
    header.threadid = start_channel;

    // generate the next VDIF header 
    nextVDIFHeader (&header, packets_per_second);
  }
}
