/***************************************************************************
 *
 *   Copyright (C) 2015 Andrew Jameson
 *   Licensed under the Academic Free License version 2.1
 *
 ***************************************************************************/

#include "spip/UDPFormatMeerKATSPEAD1k.h"
#include "spip/Time.h"

#include "spead2/common_defines.h"
#include "spead2/common_endian.h"
#include "spead2/recv_packet.h"
#include "spead2/recv_utils.h"

#include <cstdlib>
#include <cstdio>
#include <cmath>
#include <iostream>
#include <iomanip>
#include <bitset>
#include <stdexcept>

#include <unistd.h>

#define MKBF1K64A

using namespace std;

template<typename T>
static inline T extract_bits(T value, int first, int cnt)
{
    assert(0 <= first && first + cnt <= 8 * sizeof(T));
    assert(cnt > 0 && cnt < 8 * sizeof(T));
    return (value >> first) & ((T(1) << cnt) - 1);
}

spip::UDPFormatMeerKATSPEAD1k::UDPFormatMeerKATSPEAD1k()
{
  packet_header_size = 56 + 8;
#ifdef MKBF1K64A
  packet_data_size   = 1024;
#else
  packet_data_size   = 4096;
#endif
  packet_size = packet_header_size + packet_data_size;

  obs_start_sample = 0;
  npol = 1;
  ndim = 2;
  nbit = 8;
  nsamp_per_heap = 256;
  nbytes_per_samp = (ndim * npol * nbit) / 8;

  // nchan is variable depending on number of active paritions
  nchan = 1024;
  nbytes_per_heap = nsamp_per_heap * nchan * nbytes_per_samp; 
  samples_to_byte_offset = 1;
  heap_size = nbytes_per_heap;

  // this is the average size 
#ifdef MKBF1K64A
  avg_pkt_size = 1024;
#else
  avg_pkt_size = 4096;
#endif
  pkts_per_heap = (unsigned) ceil ( (float) (nsamp_per_heap * nchan * nbytes_per_samp) / (float) avg_pkt_size);

  first_heap = true;
  first_packet = false;
}

spip::UDPFormatMeerKATSPEAD1k::~UDPFormatMeerKATSPEAD1k()
{
}

void spip::UDPFormatMeerKATSPEAD1k::configure(const spip::AsciiHeader& config, const char* suffix)
{
  if (config.get ("NPOL", "%u", &header_npol) != 1)
    throw invalid_argument ("NPOL did not exist in config");
  if (config.get ("START_CHANNEL", "%u", &start_channel) != 1)
    throw invalid_argument ("START_CHANNEL did not exist in config");
  if (config.get ("END_CHANNEL", "%u", &end_channel) != 1)
    throw invalid_argument ("END_CHANNEL did not exist in config");
  if (config.get ("TSAMP", "%lf", &tsamp) != 1)
    throw invalid_argument ("TSAMP did not exist in config");
  if (config.get ("ADC_SAMPLE_RATE", "%lu", &adc_sample_rate) != 1)
    throw invalid_argument ("ADC_SAMPLE_RATE did not exist in config");
  if (config.get ("BW", "%lf", &bw) != 1)
    throw invalid_argument ("BW did not exist in config");

  nchan_per_stream = 1;
  nchan = (end_channel - start_channel) + 1;
  nstream = nchan / nchan_per_stream;
  nbytes_per_heap = nsamp_per_heap * nchan * nbytes_per_samp;
  samples_to_byte_offset = (double) (bw * 1e6 * ndim * header_npol) / adc_sample_rate; 
  heap_size = nbytes_per_heap / nstream;

  configured = true;
}

void spip::UDPFormatMeerKATSPEAD1k::prepare (spip::AsciiHeader& header, const char * suffix)
{
  char * key = (char *) malloc (128);
  char * val = (char *) malloc (128);

  if (strcmp(suffix, "_0") == 0)
    offset = 0;
  else
    offset = 1;

  if (header.get ("NCHAN_PER_STREAM", "%u", &nchan_per_stream) != 1)
    throw invalid_argument ("NCHAN_PER_STREAM did not exist in header");
  nstream = nchan / nchan_per_stream;
  half_nstream = nstream / 2;
  
#ifdef _DEBUG
  cerr << "spip::UDPFormatMeerKATSPEAD1k::prepare nstream=" << nstream 
       << " half_nstream=" << half_nstream << endl;
#endif

  heap_size = nbytes_per_heap / nstream;

  if (header.get ("ADC_SYNC_TIME", "%ld", &adc_sync_time) != 1)
    throw invalid_argument ("ADC_SYNC_TIME did not exist in config");
  if (header.get ("UTC_START", "%s", key) != 1)
    throw invalid_argument ("UTC_START did not exist in header");

  spip::Time utc_start(key);

#ifdef _DEBUG
  cerr << "adc_sync_time=" << adc_sync_time << endl;
  cerr << "utc_start.get_time=" << utc_start.get_time() << endl;
  cerr << "adc_sample_rate=" << adc_sample_rate << endl;

  // offset from ADC sync
  cerr << "OFFSET_TIME=" << (utc_start.get_time() - adc_sync_time) << " ADC_SYNC_TIME=" << adc_sync_time << " UTC_START_TIME=" << utc_start.get_time() << endl;
#endif

  // the start sample (at ADC_SAMPLE_RATE), relative to sync time  for the exact UTC start second
  obs_start_sample = (int64_t) (adc_sample_rate * (utc_start.get_time() - adc_sync_time));

  // sample rate of PFB/CBF
  double sample_rate = double(1e6) / tsamp;

  // number of ADC samples per PFB sample
  uint64_t adc_samples_per_sample = uint64_t(rint(double(adc_sample_rate) / sample_rate));

  // number of ADC samples per PFB heap - this should always be 524288
  adc_samples_per_heap = adc_samples_per_sample * nsamp_per_heap;
  if (adc_samples_per_heap != 524288)
    throw invalid_argument("ADC samples per heap != 524288");

#ifdef _DEBUG
  cerr << "OBS_START_SAMPLE=" << obs_start_sample << " ADC_SAMPLES_PER_SAMPLE=" << adc_samples_per_sample << endl;
#endif

  // observation should begin on first heap after UTC_START, add offset in picoseconds to header
  int64_t picoseconds = 0;
  int64_t modulus = obs_start_sample % adc_samples_per_heap;
  if (modulus > 0)
  {
    int64_t adc_samples_to_add = adc_samples_per_heap - modulus;   
    obs_start_sample += adc_samples_to_add;
    double offset_seconds = double(adc_samples_to_add) / double(adc_sample_rate);
    uint64_t offset_picoseconds = uint64_t(rintf(offset_seconds * 1e12));
#ifdef _DEBUG
    cerr << "obs_start_sample=" << obs_start_sample << " modulus=" << modulus << " adc_samples_to_add=" << adc_samples_to_add << " offset_picoseconds=" << offset_picoseconds << endl;
    cerr << "MODULUS=" << modulus << " PICOSECONDS=" << offset_picoseconds << endl;
#endif
    picoseconds += offset_picoseconds;
  }

  // apply the MeerKAT Precise Time offset
  double precise_time_fraction_nanoseconds = 0;
#ifdef MEERKAT_BOTH_PT_SENSORS
  double precise_time_fraction_polh = 0;
  double precise_time_fraction_polv = 0;
  if (header.get ("PRECISETIME_FRACTION_POLH", "%lf", &precise_time_fraction_polh) != 1)
    cerr << "PRECISETIME_FRACTION_POLH did not exist in header" << endl;
  if (header.get ("PRECISETIME_FRACTION_POLV", "%lf", &precise_time_fraction_polv) != 1)
    cerr << "PRECISETIME_FRACTION_POLV did not exist in header" << endl;

  // check that both sensors are non zero
  unsigned precise_time_fraction_count = 0;

  if (fabs(precise_time_fraction_polh) > 0)
  {
    precise_time_fraction_nanoseconds += precise_time_fraction_polh;
    precise_time_fraction_count++;
  }
  if (fabs(precise_time_fraction_polv) > 0)
  {
    precise_time_fraction_nanoseconds += precise_time_fraction_polv;
    precise_time_fraction_count++;
  }

  // if both sensors are good
  if (precise_time_fraction_count == 2)
  {
    precise_time_fraction_nanoseconds /= 2;
    double difference = fabs(precise_time_fraction_polh - precise_time_fraction_polv);
    double adc_sampling_time_ns = double(1e9) / double(adc_sample_rate);
    if (difference > 32 * adc_sampling_time_ns)
      cerr << "Warning: difference between precisetime sensors " 
           << difference << " > 32 ADC samples" << endl;
  }
  else if (precise_time_fraction_count == 1)
  {
    cerr << "Warning: one of the precise time sensors was zero: "
         << "  polh=" << precise_time_fraction_polh
         << "  polv=" << precise_time_fraction_polv << endl;
  }
  else
    cerr << "Warning: both precise time sensors were zero" << endl;

  header.set ("PRECISETIME_FRACTION_AVG", "%lf", precise_time_fraction_nanoseconds);

#else

  // 6-Apr-2018
  // Thomas Abbott advised only to use Vpol sensor
  double precise_time_fraction_polv = 0;
  if (header.get ("PRECISETIME_FRACTION_POLV", "%lf", &precise_time_fraction_polv) != 1)
    cerr << "PRECISETIME_FRACTION_POLV did not exist in header" << endl;
  precise_time_fraction_nanoseconds = precise_time_fraction_polv;

  header.set ("PRECISETIME_FRACTION", "%lf", precise_time_fraction_nanoseconds);

#endif

  int64_t precise_time_fraction_picoseconds = int64_t(precise_time_fraction_nanoseconds * 1e3);

#ifdef _DEBUG
  cerr << "precise_time_fraction avg=" << precise_time_fraction_picoseconds << endl;
#endif
  // AJ TODO remove
  //picoseconds += precise_time_fraction_picoseconds;
#ifdef _DEBUG
  cerr << "PICOSECONDS=" << picoseconds << endl;
#endif
  header.set ("PICOSECONDS", "%ld", picoseconds);

#ifdef _DEBUG
  cerr << "UTC_START=" << key<< " obs_start_sample=" << obs_start_sample << " modulus=" 
       << modulus << " picoseconds=" << picoseconds <<  endl;
#endif

  free (key);
  free (val);

  prepared = true;
}

void spip::UDPFormatMeerKATSPEAD1k::conclude ()
{
}


void spip::UDPFormatMeerKATSPEAD1k::generate_signal ()
{
}

uint64_t spip::UDPFormatMeerKATSPEAD1k::get_samples_for_bytes (uint64_t nbytes)
{
#ifdef _DEBUG
  cerr << "spip::UDPFormatMeerKATSPEAD1k::get_samples_for_bytes npol=" << npol 
       << " ndim=" << ndim << " nchan=" << nchan << endl;
#endif
  uint64_t nsamps = nbytes / (npol * ndim * nchan);
  return nsamps;
}

uint64_t spip::UDPFormatMeerKATSPEAD1k::get_resolution ()
{
  return heap_size * nstream;
}

void spip::UDPFormatMeerKATSPEAD1k::set_channel_range (unsigned start, unsigned end) 
{
  cerr << "spip::UDPFormatMeerKATSPEAD1k::set_channel_range start=" << start 
       << " end=" << end << endl;
  start_channel = start;
  end_channel   = end;
  nchan = (end - start) + 1;
  cerr << "spip::UDPFormatMeerKATSPEAD1k::set_channel_range nchan=" <<  nchan << endl;
}

inline void spip::UDPFormatMeerKATSPEAD1k::encode_header_seq (char * buf, uint64_t seq)
{
  encode_header (buf);
}

inline void spip::UDPFormatMeerKATSPEAD1k::encode_header (char * buf)
{
  //memcpy (buf, (void *) &header, sizeof (meerkat_spead_udp_hdr_t));
}


inline std::size_t spip::UDPFormatMeerKATSPEAD1k::decode_cbf_packet (spip::cbf_packet_header &out, const uint8_t *data, std::size_t max_size)
{
  std::uint64_t header = spead2::load_be<std::uint64_t>(data);
#ifdef _DEBUG
  if (extract_bits(header, 48, 16) != magic_version)
  {
    cerr << "packet rejected because magic or version did not match"  << endl;
    return 0;
  }
  int item_id_bits = extract_bits(header, 40, 8) * 8;
#endif
  int heap_address_bits = extract_bits(header, 32, 8) * 8;
#ifdef _DEBUG
  if (item_id_bits == 0 || heap_address_bits == 0)
  {
    cerr << "packet rejected because flavour is invalid" << endl;
    return 0;
  }
#endif

#ifdef _DEBUG
  if (item_id_bits + heap_address_bits != 8 * sizeof(spead2::item_pointer_t))
  {
    cerr << "packet rejected because flavour is not SPEAD-64-*" << endl;;
    return 0;
  }
#endif

  out.n_items = extract_bits(header, 0, 16);
#ifdef _DEBUG
  if (std::size_t(out.n_items) * sizeof(spead2::item_pointer_t) + 8 > max_size)
  {
    cerr << "packet rejected because the items overflow the packet" << endl;
    return 0;
  }
#endif

  // Mark specials as not found
  out.heap_cnt = -1;
  out.heap_length = -1;
  out.payload_offset = -1;
  out.payload_length = -1;
  out.timestamp = -1;
  out.channel = -1;

  // Look for special items
  spead2::recv::pointer_decoder decoder(heap_address_bits);
  int first_regular = out.n_items;
  for (int i = 0; i < out.n_items; i++)
  {
    spead2::item_pointer_t pointer = spead2::load_be<spead2::item_pointer_t>(data + 8 + i * sizeof(spead2::item_pointer_t));
    bool special;
    if (decoder.is_immediate(pointer))
    {
      special = true;
      switch (decoder.get_id(pointer))
      {
        case spead2::HEAP_CNT_ID:
          out.heap_cnt = decoder.get_immediate(pointer);
          break;
        case spead2::HEAP_LENGTH_ID:
          out.heap_length = decoder.get_immediate(pointer);
          break;
        case spead2::PAYLOAD_OFFSET_ID:
          out.payload_offset = decoder.get_immediate(pointer);
          break;
        case spead2::PAYLOAD_LENGTH_ID:
          out.payload_length = decoder.get_immediate(pointer);
          break;
        case 0x1600:
          out.timestamp = decoder.get_immediate(pointer);
          break;
        case 0x4103:
          out.channel = decoder.get_immediate(pointer);
          break;
        default:
          special = false;
          break;
      }
    }
    else
      special = false;
      if (!special)
          first_regular = std::min(first_regular, i);
  }

#ifdef _DEBUG
  if (out.heap_cnt == -1 || out.payload_offset == -1 || out.payload_length == -1)
  {
    cerr << "packet rejected because it does not have required items" << endl;
    return 0;
  }
#endif

  std::size_t size = out.payload_length + out.n_items * sizeof(spead2::item_pointer_t) + 8;

#ifdef _DEBUG
  if (size > max_size)
  {
    cerr << "packet rejected because payload length [" << size 
         << "] overflows packet size [" << max_size << "]" << endl;
    return 0;
  }
  if (out.heap_length >= 0 && out.payload_offset + out.payload_length > out.heap_length)
  {
    cerr << "packet rejected because payload would overflow given heap length" << endl;
    return 0;
  }
#endif

  // Adjust the pointers to skip the specials, since live_heap::add_packet does not
  // need them
  out.pointers = data + 8 + first_regular * sizeof(spead2::item_pointer_t);
  out.n_items -= first_regular;
  out.payload = out.pointers + out.n_items * sizeof(spead2::item_pointer_t);
  out.heap_address_bits = heap_address_bits;
  return size;
}

inline int64_t spip::UDPFormatMeerKATSPEAD1k::decode_packet (char* buf, unsigned * pkt_size)
{
#ifdef MKBF1K64A
  decode_cbf_packet (cbf_header, (const uint8_t *) buf, 1088);
#else
  decode_cbf_packet (cbf_header, (const uint8_t *) buf, 4160);
#endif

  *pkt_size = (unsigned) cbf_header.payload_length;

  if (!prepared || !configured)
    throw runtime_error ("Cannot process packet if not configured and prepared");

  // if this packet is a valid CBF packet
  if (cbf_header.n_items == 1 && cbf_header.heap_length == heap_size)
  {
    // determine the spead stream for this packet
    spead_stream = (cbf_header.channel - start_channel) / nchan_per_stream;

    // check the spead stream is valid
    if ((spead_stream < 0) || (spead_stream >= nstream))
      return -1;

    // determine the sample relative to start
    const int64_t obs_sample = cbf_header.timestamp - obs_start_sample;
   
#ifdef _DEBUG
    if (!first_packet)
    {
      if (offset == 1)
        cerr << "FIRST PACKET stream=" << spead_stream << " cbf_header.timestamp=" << cbf_header.timestamp
              << " obs_start_sample=" << obs_start_sample
            << " obs_sample=" << obs_sample
             << " samples_to_byte_offset=" << samples_to_byte_offset
             << " heap_offset=" << (int64_t) (obs_sample * samples_to_byte_offset) << endl;
       first_packet = true;
    }
#endif

    // if this packet pre-dates our start time, ignore
    if (obs_sample < 0)
     return -1;

    // compute the byte offset for this stream
    return uint64_t(obs_sample * samples_to_byte_offset) + 
                   (spead_stream * cbf_header.heap_length) +
                    cbf_header.payload_offset;
  } 
  else
  {
    //print_cbf_packet_header();
    if (check_stream_stop ())
      return -2;
    else
      return -1;
  }
}

inline int64_t spip::UDPFormatMeerKATSPEAD1k::get_subband (int64_t byte_offset, int nsubband)
{
  if (spead_stream >= half_nstream)
    return 1;
  else
    return 0;
}

inline int spip::UDPFormatMeerKATSPEAD1k::insert_last_packet (char * buffer)
{
  memcpy (buffer, cbf_header.payload, cbf_header.payload_length);
  return 0;
}

// generate the next packet in the cycle
inline void spip::UDPFormatMeerKATSPEAD1k::gen_packet (char * buf, size_t bufsz)
{
  // cycle through each of the channels to produce a packet with 1024 
  // time samples and two polarisations

  // write the new header
  encode_header (buf);

  /*
  // increment channel number
  header.frequency_channel.item_address++;
  if (header.frequency_channel.item_address > end_channel)
  {
    header.frequency_channel.item_address = start_channel;
    header.heap_number.item_address++;
    //nsamp_offset += header.nsamp;
  }
  */

}

bool spip::UDPFormatMeerKATSPEAD1k::check_stream_stop ()
{
  spead2::recv::pointer_decoder decoder(cbf_header.heap_address_bits);
  for (int i = 0; i < cbf_header.n_items; i++)
  {
    spead2::item_pointer_t pointer = spead2::load_be<spead2::item_pointer_t>(cbf_header.pointers + i * sizeof(spead2::item_pointer_t));
    spead2::s_item_pointer_t item_id = decoder.get_id(pointer);
    if (item_id == spead2::STREAM_CTRL_ID && decoder.is_immediate(pointer) &&
        decoder.get_immediate(pointer) == spead2::CTRL_STREAM_STOP)
      return true;
  }
  return false;
}

void spip::UDPFormatMeerKATSPEAD1k::print_packet_header()
{
  cerr << "heap_cnt=" << cbf_header.heap_cnt << " heap_length=" << cbf_header.heap_length
       << " payload_offset=" << cbf_header.payload_offset
       << " payload_length=" << cbf_header.payload_length << " n_items="
       << cbf_header.n_items << endl;

  spead2::recv::pointer_decoder decoder(cbf_header.heap_address_bits);
  for (int i = 0; i < cbf_header.n_items; i++)
  {
    // there should be items 
    spead2::item_pointer_t pointer = spead2::load_be<spead2::item_pointer_t>(cbf_header.pointers + i * sizeof(spead2::item_pointer_t));

    spead2::s_item_pointer_t item_id = decoder.get_id(pointer);
    if (decoder.is_immediate(pointer))
    {
      uint64_t value = decoder.get_immediate(pointer);
      cerr << "item[" << i << "] is_immediate " << " item_id=" << item_id << " value=" << value << endl;
    }
    else
      cerr << "item[" << i << "] item_id=" << item_id << endl;

  }
}
