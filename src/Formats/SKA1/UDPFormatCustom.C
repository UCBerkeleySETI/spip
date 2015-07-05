/***************************************************************************
 *
 *   Copyright (C) 2015 Andrew Jameson
 *   Licensed under the Academic Free License version 2.1
 *
 ***************************************************************************/

#include "spip/UDPFormatCustom.h"

#include <cstdlib>
#include <iostream>

using namespace std;

spip::UDPFormatCustom::UDPFormatCustom()
{
  packet_header_size = sizeof(ska1_custom_udp_header_t);
  packet_data_size   = 4096;

  nsamp_offset = 0;

  header.cbf_version = 1;
  header.seq_number = 0;
  header.weights = 1;
  header.nsamp = spip::UDPFormatCustom::get_samples_per_packet();
  header.channel_number = 0;
  header.beam_number = 0;
}

spip::UDPFormatCustom::~UDPFormatCustom()
{
  cerr << "spip::UDPFormatCustom::~UDPFormatCustom()" << endl;
}

void spip::UDPFormatCustom::generate_signal ()
{
}

uint64_t spip::UDPFormatCustom::get_samples_for_bytes (uint64_t nbytes)
{
  cerr << "npol=" << npol << " ndim=" << ndim << " nchan=" << nchan << endl;
  uint64_t nsamps = nbytes / (npol * ndim * nchan);
  return nsamps;
}

void spip::UDPFormatCustom::set_channel_range (unsigned start, unsigned end) 
{
  start_channel = start;
  end_channel   = end;
  nchan = (end - start) + 1;
  header.channel_number = start;
}

inline void spip::UDPFormatCustom::encode_header_seq (char * buf, size_t bufsz, uint64_t seq)
{
  header.seq_number = seq;
  encode_header (buf, bufsz);
}

inline void spip::UDPFormatCustom::encode_header (char * buf, size_t bufsz)
{
  memcpy (buf, (void *) &header, sizeof (ska1_custom_udp_header_t));
}

inline uint64_t spip::UDPFormatCustom::decode_header_seq (char * buf, size_t bufsz)
{
  memcpy ((void *) &header, buf, sizeof(uint64_t));
  return header.seq_number;
}

inline void spip::UDPFormatCustom::decode_header (char * buf, size_t bufsz)
{
  memcpy ((void *) &header, buf, bufsz);
}

inline int spip::UDPFormatCustom::insert_packet (char * buf, char * pkt, uint64_t start_samp, uint64_t next_start_samp)
{
  const uint64_t sample_number = header.seq_number * 1024; 
  if (sample_number < start_samp)
  {
    cerr << "header.seq_number=" << header.seq_number << " header.channel_number=" << header.channel_number << endl;
    cerr << "sample_number=" << sample_number << " start_samp=" << start_samp << endl;
    return 2;
  }
  if (sample_number >= next_start_samp)
  {
    return 1;
  }
 
  // determine the channel offset in bytes
  const unsigned offset = (header.channel_number - start_channel) * channel_stride;
  const unsigned sample_offset = header.seq_number - start_samp;

  // incremement buf pointer to 
  const unsigned pol0_offset = offset + sample_offset;
  const unsigned pol1_offset = pol0_offset + chanpol_stride;

  memcpy (buf + pol0_offset, pkt, 2048);
  memcpy (buf + pol1_offset, pkt + 2048, 2048);

  return 0;
}

// generate the next packet in the cycle
inline void spip::UDPFormatCustom::gen_packet (char * buf, size_t bufsz)
{
  // cycle through each of the channels to produce a packet with 1024 
  // time samples and two polarisations

  // write the new header
  encode_header (buf, bufsz);

  // increment channel number
  header.channel_number++;
  if (header.channel_number >= end_channel)
  {
    header.channel_number = start_channel;
    header.seq_number++;
    nsamp_offset += header.nsamp;
  }

}

void spip::UDPFormatCustom::print_packet_header()
{
  cerr << "seq=" << header.seq_number << " chan=" << header.channel_number << endl;
}