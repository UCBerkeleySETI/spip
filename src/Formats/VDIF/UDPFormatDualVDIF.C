/***************************************************************************
 *
 *   Copyright (C) 2018 Andrew Jameson
 *   Licensed under the Academic Free License version 2.1
 *
 ***************************************************************************/

//#define _DEBUG

#include "spip/UDPFormatDualVDIF.h"
#include "spip/Time.h"

#include <cstdio>
#include <cstdlib>
#include <iostream>
#include <stdexcept>

using namespace std;

spip::UDPFormatDualVDIF::UDPFormatDualVDIF(int pps) : UDPFormatVDIF (pps)
{
#ifdef _DEBUG
  cerr << "spip::UDPFormatDualVDIF::UDPFormatDualVDIF()" << endl;
#endif
}

spip::UDPFormatDualVDIF::~UDPFormatDualVDIF()
{
#ifdef _DEBUG
  cerr << "spip::UDPFormatDualVDIF::~UDPFormatDualVDIF()" << endl;
#endif
}

uint64_t spip::UDPFormatDualVDIF::get_resolution ()
{
#ifdef _DEBUG
  cerr << "spip::UDPFormatDualVDIF::get_resolution() " << header_npol * packet_data_size << endl;
#endif
  return header_npol * packet_data_size;
}

inline int64_t spip::UDPFormatDualVDIF::decode_packet (char * buf, unsigned * pkt_size)
{
  // header is stored in the front of the packet
  vdif_header * header_ptr = (vdif_header *) buf;

  // configure the stream using the first packet
  if (!configured_stream)
  {
    configure_stream (buf);
  }

  // this UDP format assumes N threads, for either pol 0, 1 or 2
  int thread_id = getVDIFThreadID (header_ptr);

  if (thread_id != 0 && thread_id != 1 && thread_id != 2)
    throw invalid_argument ("VDIF thread ID must be 0, 1 or 2");

  // the size of a VDIF UDP packet
  *pkt_size = packet_data_size;

  // adjust the pointr to the packet data frame
  payload = buf + packet_header_size;

  // extract key parameters from the header
  const int offset_second = getVDIFFrameEpochSecOffset (header_ptr) - start_second;
  const int frame_number  = getVDIFFrameNumber (header_ptr);

  // calculate the byte offset for this frame within the data stream
  int64_t byte_offset = (offset_second * bytes_per_second) + (frame_number * frame_size) + (thread_id * packet_data_size);

#ifdef _DEBUG
  cerr << "spip::UDPFormatDualVDIF::decode_packet thread_id =" << thread_id  
       << " offset_second=" << offset_second << " frame_number=" << frame_number 
       << " byte_offset=" << byte_offset << endl;
#endif

  return (int64_t) byte_offset;
}

// generate the next packet in the sequence
inline void spip::UDPFormatDualVDIF::gen_packet (char * buf, size_t bufsz)
{
  // header is stored in the front of the packet
  vdif_header * header_ptr = (vdif_header *) buf;

  // get the current thread ID and increment
  int thread_id = getVDIFThreadID (header_ptr) + 1;

  if (unsigned(thread_id) > header_npol)
  {
    thread_id = 0;
    setVDIFThreadID (header_ptr, thread_id);

    // generate the next VDIF header 
    nextVDIFHeader (&header, packets_per_second);
  }
}
