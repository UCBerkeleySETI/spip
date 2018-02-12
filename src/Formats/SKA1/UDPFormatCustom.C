/***************************************************************************
 *
 *   Copyright (C) 2015 Andrew Jameson
 *   Licensed under the Academic Free License version 2.1
 *
 ***************************************************************************/

#include "spip/UDPFormatCustom.h"

#include <cstdlib>
#include <iostream>
#include <stdexcept>

using namespace std;

spip::UDPFormatCustom::UDPFormatCustom()
{
  packet_header_size = sizeof(ska1_custom_udp_header_t);

  // defaults, these must be re-read at configure time
  nsamp_per_packet = 256;
  nsamp_per_weight = 256;
  nchan_per_packet = 8;

  packet_weights_size = nchan_per_packet * nsamp_per_packet / nsamp_per_weight;
  packet_data_size = nchan_per_packet * nsamp_per_packet * nbit * UDP_FORMAT_CUSTOM_NDIM * UDP_FORMAT_CUSTOM_NPOL;
  packet_size = packet_header_size + packet_weights_size + packet_data_size;

  ncfcbi = 1;
  nchan = 8;

  header.cbf_version = 1;
  header.packet_sequence_number = 0;
  header.seconds_from_epoch = 0;
  header.attoseconds_from_integer = 0;
  header.cfcbi_number = 0;
  header.beam_number = 0;
}

spip::UDPFormatCustom::~UDPFormatCustom()
{
  cerr << "spip::UDPFormatCustom::~UDPFormatCustom()" << endl;
}

void spip::UDPFormatCustom::configure(const spip::AsciiHeader& config, const char* suffix)
{
  cerr << "spip::UDPFormatCustom::configure()" << endl; 
  double tsamp;

  if (config.get ("START_CHANNEL", "%u", &start_channel) != 1)
    throw invalid_argument ("START_CHANNEL did not exist in header");
  if (config.get ("END_CHANNEL", "%u", &end_channel) != 1)
    throw invalid_argument ("END_CHANNEL did not exist in header");
  if (config.get ("WT_NSAMP", "%u", &nsamp_per_weight) != 1)
    throw invalid_argument ("WT_NSAMP did not exist in header");
  if (config.get ("UDP_NSAMP", "%u", &nsamp_per_packet) != 1)
    throw invalid_argument ("UDP_NSAMP did not exist in header");
  if (config.get ("UDP_NCHAN", "%u", &nchan_per_packet) != 1)
    throw invalid_argument ("UDP_NCHAN did not exist in header");
  if (config.get ("TSAMP", "%lf", &tsamp) != 1)
    throw invalid_argument ("TSAMP did not exist in header");

#ifdef _DEBUG
  cerr << "spip::UDPFormatCustom::configure nsamp_per_weight=" << nsamp_per_weight << endl;
  cerr << "spip::UDPFormatCustom::configure nsamp_per_packet=" << nsamp_per_packet<< endl;
  cerr << "spip::UDPFormatCustom::configure nchan_per_packet=" << nchan_per_packet<< endl;
#endif

  packet_weights_size = (nchan_per_packet * nsamp_per_packet * sizeof(float)) / nsamp_per_weight;
  packet_data_size = (nchan_per_packet * nsamp_per_packet * nbit * UDP_FORMAT_CUSTOM_NDIM * UDP_FORMAT_CUSTOM_NPOL) / 8;

  packet_size = packet_header_size + packet_weights_size + packet_data_size;

#ifdef _DEBUG
  cerr << "spip::UDPFormatCustom::configure packet_header_size=" << packet_header_size  << endl;
  cerr << "spip::UDPFormatCustom::configure packet_weights_size=" << packet_weights_size  << endl;
  cerr << "spip::UDPFormatCustom::configure packet_data_size=" << packet_data_size << endl;
#endif

  // number of channels in this data stream
  nchan = (end_channel - start_channel) + 1;

  // configured CFCBI's in this data stream
  start_cfcbi = start_channel / nchan_per_packet;
  end_cfcbi = end_channel / nchan_per_packet;
  ncfcbi = (end_cfcbi - start_cfcbi) + 1;

#ifdef _DEBUG
  cerr << "spip::UDPFormatCustom::configure start_cfcbi=" << start_cfcbi << " end_cfcbi=" << end_cfcbi << " ncfcbi=" << ncfcbi << endl;
#endif

  // the number of attoseconds that increment with each packet
  uint64_t attoseconds_per_microsecond = 1e12;
  attoseconds_per_packet = attoseconds_per_microsecond * nsamp_per_packet * tsamp;

  // other defaults
  header.cbf_version = 1;
  header.packet_sequence_number = 0;
  header.seconds_from_epoch = 0;
  header.attoseconds_from_integer = 0;
  header.cfcbi_number = start_cfcbi;
  header.beam_number = 0;
}

void spip::UDPFormatCustom::prepare (spip::AsciiHeader& header, const char * suffix)
{
#ifdef _DEBUG
  cerr << "spip::UDPFormatCustom::prepare()" << endl; 
#endif

  // number of bytes in single packet CFCBI
  cfcbi_stride = (nsamp_per_packet * nchan_per_packet * ndim * npol * nbit) / 8;

  // number of bytes in a "sequence number"
  seq_stride = cfcbi_stride * ncfcbi;
}

uint64_t spip::UDPFormatCustom::get_samples_for_bytes (uint64_t nbytes)
{
#ifdef _DEBUG
  cerr << "spip::UDPFormatCustom::get_samples_for_bytes npol=" << npol 
       << " ndim=" << ndim << " nchan=" << nchan << endl;
#endif
  uint64_t nsamps = nbytes / (npol * ndim * nchan * nbit / 8);
  return nsamps;
}

uint64_t spip::UDPFormatCustom::get_resolution ()
{
  uint64_t nbits = nsamp_per_packet * nchan_per_packet * ndim * npol * nbit * ncfcbi;
  return nbits / 8;
}

inline void spip::UDPFormatCustom::encode_header_seq (char * buf, uint64_t seq)
{
  header.packet_sequence_number = seq;
  encode_header (buf);
}

inline void spip::UDPFormatCustom::encode_header (char * buf)
{
  memcpy (buf, (void *) &header, sizeof (ska1_custom_udp_header_t));
}

// decode JUST the packet_sequence_number
inline uint64_t spip::UDPFormatCustom::decode_header_seq (char * buf)
{
  ska1_custom_udp_header_t * ptr = (ska1_custom_udp_header_t *) buf;
  return ptr->packet_sequence_number;
}

inline unsigned spip::UDPFormatCustom::decode_header (char * buf)
{
  header_ptr = (ska1_custom_udp_header_t *) buf;
  //memcpy ((void *) &header, buf, sizeof(header));
  return packet_data_size;
}

inline int64_t spip::UDPFormatCustom::decode_packet (char *buf, unsigned * payload_size)
{
  // copy the header from the packet to private buffer
  *payload_size = decode_header (buf);

  // set the pointer to the weights
  weights_ptr = buf + packet_header_size;

  // set the pointer to the payload
  payload_ptr = weights_ptr + packet_weights_size;

#ifdef _DEBUG
  cerr << "spip::UDPFormatCustom::decode_packet header.channel_number=" << header.channel_number 
       << " start_channel=" << start_channel 
       << " channel_stride=" << channel_stride << endl;
#endif

  // compute absolute byte offset for this packet within the data stream
  //const uint64_t byte_offset = (header.packet_sequence_number * seq_stride) + 
  //                             ((header.cfcbi_number - start_cfcbi) * cfcbi_stride);
  const uint64_t byte_offset = (header_ptr->packet_sequence_number * seq_stride) + 
                               ((header_ptr->cfcbi_number - start_cfcbi) * cfcbi_stride);
  return (int64_t) byte_offset;  
}

inline int spip::UDPFormatCustom::insert_last_packet (char * buffer)
{
  memcpy (buffer, payload_ptr, packet_data_size);
  return 0;
}

// generate the next packet in the cycle
inline void spip::UDPFormatCustom::gen_packet (char * buf, size_t bufsz)
{
  // write the new header
  encode_header (buf);

  // increment CFCBI number
  header.cfcbi_number++;

  if (header.cfcbi_number > end_cfcbi)
  {
    header.cfcbi_number = start_cfcbi;
    header.packet_sequence_number++;

    // update the packet timestamps accordingly
    header.attoseconds_from_integer += attoseconds_per_packet;
    while (header.attoseconds_from_integer >= 1e18)
    {
      header.seconds_from_epoch += 1;
      header.attoseconds_from_integer -= 1e18;
    }
  }
}

void spip::UDPFormatCustom::print_packet_header()
{
  cerr << "packet_sequence_number=" << header.packet_sequence_number 
       << " cfcbi_number=" << header.cfcbi_number << endl;
}
