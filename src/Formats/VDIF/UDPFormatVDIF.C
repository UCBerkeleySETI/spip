/***************************************************************************
 *
 *   Copyright (C) 2015 Andrew Jameson
 *   Licensed under the Academic Free License version 2.1
 *
 ***************************************************************************/

//#define _DEBUG

#include "spip/UDPFormatVDIF.h"
#include "spip/Time.h"
#include "spip/Error.h"

#include <cstdio>
#include <cstdlib>
#include <iostream>
#include <stdexcept>

using namespace std;

spip::UDPFormatVDIF::UDPFormatVDIF(int pps)
{
#ifdef DEBUG
  cerr << "spip::UDPFormatVDIF::UDPFormatVDIF(" << pps << ")" << endl;
#endif

  // this assumes the use of the 32 byte VDIF Data Frame Header
  packet_header_size = VDIF_HEADER_BYTES;
  // assume the best case scenario for now
  packet_data_size   = 8192;

  // each VDIF thread only contains 1 polarisation
  npol = 1;

  packets_per_second = pps;
  tsamp = 0;
  bw = 0;
  start_channel = 0;
  end_channel = 0;
  thread_id = 0;

  nsamp_per_packet = 0;
  bytes_per_second = 0;
  prev_frame_number = 0;

  // we will "extract" the UTC_START from the data stream
  self_start = true;
}

spip::UDPFormatVDIF::~UDPFormatVDIF()
{
#ifdef _DEBUG
  cerr << "spip::UDPFormatVDIF::~UDPFormatVDIF()" << endl;
#endif
}

void spip::UDPFormatVDIF::configure (const spip::AsciiHeader& config, const char* suffix)
{
#ifdef _DEBUG
  cerr << "spip::UDPFormatVDIF::configure()" << endl;
#endif

  if (config.get ("NCHAN", "%u", &nchan) != 1)
    throw invalid_argument ("NCHAN did not exist in config");
  if (config.get ("NDIM", "%u", &ndim) != 1)
    throw invalid_argument ("NDIM did not exist in config");
  if (config.get ("NBIT", "%u", &nbit) != 1)
    throw invalid_argument ("NBIT did not exist in config");
  if (config.get ("NPOL", "%u", &header_npol) != 1)
    throw invalid_argument ("NPOL did not exist in config");
  if (config.get ("START_CHANNEL", "%u", &start_channel) != 1)
    throw invalid_argument ("START_CHANNEL did not exist in config");
  if (config.get ("END_CHANNEL", "%u", &end_channel) != 1)
    throw invalid_argument ("END_CHANNEL did not exist in config");
  if (config.get ("TSAMP", "%lf", &tsamp) != 1)
    throw invalid_argument ("TSAMP did not exist in config");
  if (config.get ("BW", "%lf", &bw) != 1)
    throw invalid_argument ("BW did not exist in config");
  if (config.get ("BYTES_PER_SECOND", "%lu", &bytes_per_second) != 1)
    throw invalid_argument ("BYTES_PER_SECOND did not exist in config");
  if (config.get ("UDP_NSAMP", "%u", &udp_nsamp) != 1)
    throw invalid_argument ("UDP_NSAMP did not exist in config");
  if (config.get ("VDIF_THREAD_ID", "%u", &thread_id) != 1)
  {
#ifdef _DEBUG
    cerr << "VDIF_THREAD_ID did not exist in header, assuming 0" << endl;
#endif
    thread_id = 0;
  }
  if (nchan != (end_channel - start_channel) + 1)
    throw invalid_argument ("NCHAN, START_CHANNEL and END_CHANNEL were in conflict");

  // size of a udp packet payload
  packet_data_size = (udp_nsamp * npol * nchan * nbit * ndim) / 8;

  // size of an output data frame
  frame_size = packet_data_size * header_npol;

#ifdef _DEBUG
  cerr << "spip::UDPFormatVDIF::configure packet_data_size=" << packet_data_size 
       << " frame_size=" << frame_size << endl;
#endif

  configured = true;
}

void spip::UDPFormatVDIF::prepare (spip::AsciiHeader& config, const char * suffix)
{
#ifdef _DEBUG
  cerr << "spip::UDPFormatVDIF::prepare" << endl;
#endif

  // if the stream is to start on a supplied epoch, extract the 
  // UTC_START from the config
#ifdef _DEBUG
  cerr << "spip::UDPFormatVDIF::prepare self_start=" << self_start << endl;
#endif
  if (!self_start)
  {
    char * key = (char *) malloc (128);
    if (config.get ("UTC_START", "%s", key) != 1)
      throw invalid_argument ("UTC_START did not exist in config");
#ifdef _DEBUG
    cerr << "spip::UDPFormatVDIF::prepare not self starting, supplied UTC_START=" << key << endl;
#endif
    utc_start.set_time (key);
    pico_seconds = 0;
    free (key);
  }

  // polarisation identification
  if (strcmp(suffix, "_0") == 0)
  {
    offset = 0;
    if (header_npol < 1)
      throw invalid_argument("suffix if _0 requires NPOL > 0");
  }
  else if (strcmp(suffix, "_1") == 0)
  {
    offset = 1;
    if (header_npol < 2)
      throw invalid_argument("suffix if _1 requires NPOL > 1");
  }
  else if (strcmp(suffix, "_2") == 0)
  {
    offset = 2;
    if (header_npol < 3)
      throw invalid_argument("suffix if _1 requires NPOL > 2");
  }
  else
  {
    offset = 0;
  }

  // offset of this data stream within the output data stream
  frame_offset = offset * packet_data_size;

#ifdef _DEBUG
  cerr << "spip::UDPFormatVDIF::prepare suffix=" << suffix << " offset=" << offset << " frame_offset=" << frame_offset << endl;
  cerr << "spip::UDPFormatVDIF::prepare configured_stream=false" << endl;
#endif

  prepared = true;
  configured_stream = false;
}

void spip::UDPFormatVDIF::compute_header ()
{
  // set the channel number of the first channel
  setVDIFThreadID (&header, start_channel);

  // VDIF reference epoch, 6 month period since 2000
  int gm_year = utc_start.get_gm_year();
  int ref_epoch = (gm_year - 2000) * 2;
  int gm_mon  = utc_start.get_gm_month();
  if (gm_mon >= 6)
    ref_epoch++;
  ref_epoch = ref_epoch % 64;
  setVDIFEpoch (&header,  ref_epoch);

  // MJD day for the frame
  int mjd_day = utc_start.get_mjd_day();
  setVDIFFrameMJD (&header, mjd_day);

  // get the time string of the epoch
  char key[32];
  sprintf (key, "%04d-%02d-01-00:00:00", gm_year, gm_mon);
  spip::Time epoch (key);

  // determine number of seconds since the ref epoch
  start_second = int (utc_start.get_time() - epoch.get_time());
  setVDIFFrameSecond (&header, start_second);

  // always start on frame 0
  int frame_number = 0;
  setVDIFFrameNumber (&header, frame_number);

  int vdif_nchan = nchan * npol;
  setVDIFNumChannels (&header, vdif_nchan);

  // no method for this
  header.iscomplex = (ndim - 1);

  // determine largest packet size, such that there is an integer
  // number of packets_per_second
  packet_data_size = 8192;
  uint64_t nbit_per_sample = uint64_t(nbit * npol * nchan * ndim);
  uint64_t match = bytes_per_second % packet_data_size;
  //cerr << "start: packet_data_size=" << packet_data_size << " nbit_per_sample=" << nbit_per_sample << " bytes_per_second=" << bytes_per_second << " match=" << match << endl;
  while ((packet_data_size * 8) > nbit_per_sample && match != 0)
  {
    match = bytes_per_second % packet_data_size;
    packet_data_size -= 8;
    //cerr << "find: packet_data_size=" << packet_data_size << " nbit_per_sample=" << nbit_per_sample << endl;
  }

  if (match != 0)
    throw invalid_argument ("No packet size matched VDIF1.1 criteria");

  //cerr << "end: packet_data_size=" << packet_data_size << endl;
  packets_per_second = bytes_per_second / packet_data_size;

  header.nbits = (nbit - 1);

  // bytes_per_second
  int frame_length = packet_data_size + packet_header_size;
  setVDIFFrameBytes (&header, frame_length);

#ifdef _DEBUG
  //cerr << "spip::UDPFormatVDIF::prepare UTC_START=" << utc_start.get_gmtime() << endl;
  //cerr << "spip::UDPFormatVDIF::prepare EPOCH=" << epoch.get_gmtime() << endl;
  //cerr << "spip::UDPFormatVDIF::prepare start_second=" << start_second << endl;
  //cerr << "spip::UDPFormatVDIF::prepare packet_data_size=" << packet_data_size << endl;
  //cerr << " packets_per_second=" << packets_per_second << endl;
#endif
}

void spip::UDPFormatVDIF::conclude ()
{
}

uint64_t spip::UDPFormatVDIF::get_resolution ()
{
#ifdef _DEBUG
  cerr << "spip::UDPFormatVDIF::get_resolution() " << packet_data_size << endl;
#endif
  return packet_data_size;
}

uint64_t spip::UDPFormatVDIF::get_samples_for_bytes (uint64_t nbytes)
{
  uint64_t nsamps = nbytes / (header_npol * ndim * nchan);
#ifdef _DEBUG
  cerr << "spip::UDPFormatVDIF::get_samples_for_bytes nbytes=" << nbytes 
       << " nsamps=" << nsamps << endl;
#endif
  return nsamps;
}

inline void spip::UDPFormatVDIF::encode_header_seq (char * buf, uint64_t seq)
{
  vdif_header * header_ptr = (vdif_header *) buf;

  uint64_t frame_number = seq % packets_per_second;
  uint64_t frame_second = seq / packets_per_second;

  setVDIFFrameNumber (header_ptr, frame_number);
  setVDIFFrameSecond (header_ptr, frame_second);

  encode_header (buf);
}

inline void spip::UDPFormatVDIF::encode_header (char * buf)
{
  memcpy (buf, (void *) &header, VDIF_HEADER_BYTES);
}

inline void spip::UDPFormatVDIF::decode_header (char * buf)
{
  memcpy ((void *) &header, buf, VDIF_HEADER_BYTES);
}

inline int64_t spip::UDPFormatVDIF::decode_packet (char * buf, unsigned * pkt_size)
{
  // header is stored in the front of the packet
  vdif_header * header_ptr = (vdif_header *) buf;

  // configure the stream on the first packet
  if (!configured_stream)
  {
    configure_stream (buf);
    if (!configured_stream)
      return -1;
  }

  // only accept packets from the specified VDIF thread
  if (getVDIFThreadID (header_ptr) != thread_id)
    return -1;

  *pkt_size = packet_data_size;

  payload = buf + packet_header_size;

  // extract key parameters from the header
  const int offset_second = getVDIFFrameEpochSecOffset (header_ptr) - start_second;
  const int frame_number  = getVDIFFrameNumber (header_ptr);

#ifdef _DEBUG
  if (frame_number != 0 && frame_number != prev_frame_number + 1)
    cerr << thread_id << " offset_second=" << offset_second << " frame_number=" << frame_number << " prev_frame_number=" << prev_frame_number << " dropped=" << frame_number - (prev_frame_number + 1) << endl;
  prev_frame_number = frame_number;
#endif

  // calculate the byte offset for this frame within the data stream
  int64_t byte_offset = (bytes_per_second * offset_second) + (frame_number * frame_size);

  return (int64_t) byte_offset;
}

// configure the VDIF data stream, computing the start_second
void spip::UDPFormatVDIF::configure_stream (char * buf)
{
#ifdef _DEBUG
  cerr << "spip::UDPFormatVDIF::configure_stream" << endl;
#endif

  // decode the header parameters
  decode_header (buf);

#ifdef _DEBUG
  printVDIFHeader ((const struct vdif_header *) &header, VDIFHeaderPrintLevelLong);
#endif

  int vdif_epoch = getVDIFEpoch (&header);

  // ignore non VDIF packets
  if (vdif_epoch == 0)
    return;
    
  // handle older header versions
  if ((int) (header.legacymode) == 1)
    packet_header_size = 16;
  else
    packet_header_size = 32;

  if (ndim != unsigned(int(header.iscomplex) + 1))
  {
    throw Error (InvalidState, "spip::UDPFormatVDIF::configure_stream",
                 "NDIM mismatch between CONFIG [%d] and VDIF [%d]", ndim, (header.iscomplex + 1));
  }

  if (nbit != unsigned(getVDIFBitsPerSample (&header)))
  {
    throw Error (InvalidState, "spip::UDPFormatVDIF::configure_stream",
                 "NBIT mismtach between CONFIG [%d] and VDIF header [%d]", nbit, int(getVDIFBitsPerSample (&header)));
  }

  if  (nchan * npol != unsigned(getVDIFNumChannels (&header)))
  {
    throw Error (InvalidState, "spip::UDPFormatVDIF::configure_stream",
                 "NCHAN/NPOL mismtach between config and VDIF header");
  }

  packet_data_size = getVDIFFrameBytes (&header) - packet_header_size;

  int offset_second = getVDIFFrameEpochSecOffset (&header);
  int frame_number  = getVDIFFrameNumber (&header);

  // this code will not work past 2030 something
  int gm_year = 2000 + (vdif_epoch / 2);
  int gm_month = (vdif_epoch % 2) * 6 + 1;

#ifdef _DEBUG
  cerr << "spip::UDPFormatVDIF::configure_stream decode_packet gm_year=" << gm_year << endl;
  cerr << "spip::UDPFormatVDIF::configure_stream gm_month=" << gm_month << endl;
#endif

  // get the time string of the epoch
  char * key = (char *) malloc (32);
  sprintf (key, "%04d-%02d-01-00:00:00", gm_year, gm_month);
  if (self_start)
  {
    utc_start.set_time (key);
    std::string utc_str = utc_start.get_gmtime();
    utc_start.add_seconds (offset_second);

#ifdef _DEBUG
    time_t now = time(0);
    spip::Time current(now);
    cerr << "Current UTC=" << current.get_gmtime() << " packet UTC=" << utc_start.get_gmtime() << endl;
#endif

    utc_start.add_seconds (2);
    std::string localtime_str = utc_start.get_localtime();
    utc_str = utc_start.get_gmtime();

#ifdef _DEBUG
    cerr << "spip::UDPFormatVDIF::configure_stream self starting UTC_START=" << utc_str << endl;
#endif

    // set start second to be the next whole integer second
    start_second = offset_second;
    if (frame_number > 0)
      start_second++;
    pico_seconds = 0;
  }
  // determine start_seconds of this VDIF data relative to UTC_START
  else
  {
    // add the delay here...
    Time vdif_epoch (key);
    Time vdif_packet (key);
    vdif_packet.add_seconds (offset_second);
    start_second = utc_start.get_time() - vdif_epoch.get_time();
    pico_seconds = 0;
  }

#ifdef _DEBUG
  cerr << "spip::UDPFormatVDIF::configure_stream packet_data_size=" << packet_data_size
       << " packet_header_size=" << packet_header_size << endl;
  cerr << "spip::UDPFormatVDIF::configure_stream bit=" << nbit << " npol=" << npol
       << " ndim=" << ndim << endl;
  cerr << "spip::UDPFormatVDIF::configure_stream vdif_epoch=" << vdif_epoch << " offset_second=" << offset_second
      << " frame_number=" << frame_number << endl;
#endif

  configured_stream = true;
  free (key);
}

inline int spip::UDPFormatVDIF::insert_last_packet (char * buffer)
{
  memcpy (buffer, payload, packet_data_size);
  return 0;
}

// generate the next packet in the sequence
inline void spip::UDPFormatVDIF::gen_packet (char * buf, size_t bufsz)
{
  // packets are packed in TF order 
  // write the new header
  encode_header (buf);

  // generate the next VDIF header 
  nextVDIFHeader (&header, packets_per_second);
}

void spip::UDPFormatVDIF::print_packet_header()
{
  cerr << "VDIF packet header: "
       << " second=" << getVDIFFrameSecond (&header) 
       << " frame=" << getVDIFFrameNumber (&header) << endl;
}
