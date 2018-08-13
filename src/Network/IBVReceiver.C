/***************************************************************************
 *
 *   Copyright (C) 2015 Andrew Jameson
 *   Licensed under the Academic Free License version 2.1
 *
 ***************************************************************************/

//#define _DEBUG

#include "config.h"

#include "spip/Time.h"
#include "spip/AsciiHeader.h"
#include "spip/IBVReceiver.h"
#include "sys/time.h"

#include "spip/IBVReceiver.h"

#include <unistd.h>
#include <cstring>
#include <iostream>
#include <stdexcept>
#include <new>

using namespace std;

spip::IBVReceiver::IBVReceiver()
{
  keep_receiving = true;
  have_utc_start = false;
  format = NULL;
  verbose = 1;
}

spip::IBVReceiver::~IBVReceiver()
{
  if (format)
  {
    format->conclude();
    delete format;
  }
}

void spip::IBVReceiver::configure (const char * config_str)
{
  header.load_from_str (config_str);

  if (header.get ( "NCHAN", "%u", &nchan) != 1)
    throw invalid_argument ("NCHAN did not exist in header");

  if (header.get ( "NBIT", "%u", &nbit) != 1)
    throw invalid_argument ("NBIT did not exist in header");

  if (header.get ( "NPOL", "%u", &npol) != 1)
    throw invalid_argument ("NPOL did not exist in header");

  if (header.get ( "NDIM", "%u", &ndim) != 1)
    throw invalid_argument ("NDIM did not exist in header");

  if (header.get ( "TSAMP", "%lf", &tsamp) != 1)
    throw invalid_argument ("TSAMP did not exist in header");

  if (header.get ( "BW", "%f", &bw) != 1)
    throw invalid_argument ("BW did not exist in header");

  char * buffer = (char *) malloc (128);
  if (header.get ("DATA_HOST", "%s", buffer) != 1)
    throw invalid_argument ("DATA_HOST did not exist in header");
  data_host = string (buffer);
  if (header.get ("DATA_MCAST", "%s", buffer) != 1)
    data_mcast = string ();
  else
    data_mcast = string (buffer);
  if (header.get ("DATA_PORT", "%d", &data_port) != 1)
    throw invalid_argument ("DATA_PORT did not exist in header");
  free (buffer);

  if (verbose)
    cerr << "spip::IBVReceiver::configure receiving on " 
         << data_host << ":" << data_port  << endl;

  uint64_t mega = 1e6;
  bits_per_second = (mega * nchan * npol * ndim * nbit) / tsamp;
  bytes_per_second = bits_per_second / 8;

  if (!format)
    throw runtime_error ("format was not allocated");
  format->configure (header, "");

  // configure the queue
  queue->configure (1048676, format->get_packet_size());

  // open the IBV Queue 
  if (data_mcast.size() > 0)
  {
    cerr << "spip::IBVReceiver::configure queue->open_multicast (" << data_host << ", " << data_mcast << ", " << data_port << ")" << endl;
    queue->open_multicast (data_host, data_mcast, data_port);
  }
  else
  {
    if (verbose)
      cerr << "spip::IBVReceiver::configure queue->open(" << data_host << ", " << data_port << ")" << endl;
    queue->open (data_host, data_port);
  }

  // allocate requirement memory resources
  queue->allocate ();

}

void spip::IBVReceiver::prepare ()
{
  if (verbose)
    cerr << "spip::IBVReceiver::prepare()" << endl;

  stats = new UDPStats (format->get_header_size(), format->get_data_size());

  // if this format is not self starting, check for the UTC_START
  if (!format->get_self_start ())
  {
    // check if UTC_START has been set
    char * buffer = (char *) malloc (128);
    if (header.get ("UTC_START", "%s", buffer) == -1)
    {
      cerr << "spip::IBVReceiver::open no UTC_START in header" << endl;
      time_t now = time(0);
      spip::Time utc_start (now);
      utc_start.add_seconds (2);
      std::string utc_str = utc_start.get_gmtime();
      cerr << "spip::IBVReceiver::open UTC_START=" << utc_str  << endl;
      if (header.set ("UTC_START", "%s", utc_str.c_str()) < 0)
        throw invalid_argument ("failed to write UTC_START to header");
    }
    have_utc_start = true;
  }

  format->prepare (header, "");
}

void spip::IBVReceiver::set_format (spip::UDPFormat * fmt)
{
  if (format)
    delete format;
  format = fmt;
}

// receive UDP packets for the specified time at the specified data rate
void spip::IBVReceiver::receive ()
{
  if (verbose)
    cerr << "spip::IBVReceiver::receive()" << endl;

  ssize_t got;
  uint64_t nsleeps = 0;

  // virtual block, make about 128 MB
  size_t data_bufsz = nchan * ndim * npol;
  while (data_bufsz < 128*1024*1024)
    data_bufsz *= 2;

  char * block = (char *) malloc (data_bufsz);
  bool need_next_block = false;

  // block accounting 
  int64_t curr_byte_offset = 0;
  int64_t next_byte_offset = data_bufsz;

#ifdef _DEBUG
  cerr << "spip::IBVReceiver::receive [" << curr_byte_offset << " - "
       << next_byte_offset << "] (" << 0 << ")" << endl;
#endif

  // configure the overflow buffer
  overflow = new UDPOverflow();
  uint64_t overflow_bufsz = resolution * 2;
  while (overflow_bufsz <= 65536)
    overflow_bufsz *= 2;
  overflow->resize (overflow_bufsz);
  overflow_block = (char *) overflow->get_buffer();

  if (verbose)
    cerr << "spip::IBVReceiver::receive starting main loop" << endl;

  //
  spip::IBVQueue::keep_receiving = true;
  while (spip::IBVQueue::keep_receiving)
  {
    // TODO abstract this better
    int received = queue.recv_cq.poll(n_slots, wc.get());

    for (int i=0; i<received; i++)
    {
      int index = queue.wc[i].wr_id;
      if (queue.wc[i].status != IBV_WC_SUCCESS)
      {
        cerr << "Work Request failed with code " << queue.wc[i].status << endl;
      }
      else
      {
        const void *ptr = reinterpret_cast<void *>(reinterpret_cast<std::uintptr_t>(slots[index].sge.addr));
        std::size_t len = wc[i].byte_len;

        // Sanity checks
        try
        {
          packet_buffer payload = udp_from_ethernet(const_cast<void *>(ptr), len);

          if (payload.size() < packet_size) 
          {
            cerr << "Received UDP packet of " << got << " bytes, expected " << packet_size << endl;
            spip::IBVQueue::keep_receiving = false;

            bool stopped = process_packet(payload.data(), payload.size());
            if (stopped)
              return -2;
          }
        }
        catch (packet_type_error &e)
        {
          cerr << "packet_type_error " << e.what() << endl;
        }
        catch (std::length_error &e)
        {
          cerr << "length error " << e.what() << endl;
        }
      }
      qp.post_recv(&slots[index].wr);
    }
  }
}


void spip::IBVReceiver::open_block ()
{
  need_next_block = false;

  // here we would open a ring buffer

  // update absolute limits
  curr_byte_offset = next_byte_offset;
  next_byte_offset = curr_byte_offset + data_bufsz;
  overflow_maxbyte = next_byte_offset + overflow_bufsz;

  // copy any data from the overflow into the new block
  bytes_this_buf = overflow->copy_to (block);

#ifdef _DEBUG
  cerr << "spip::UDPReceiveDB::open_block filling buffer ["
       << curr_byte_offset << " - " << next_byte_offset
       << " - " << overflow_maxbyte << "] "
       << " overflow_bufsz=" << overflow->get_bufsz() << endl;
#endif

  filled_this_block = false;
}

void spip::IBVReciver::close_block ()
{
  filled_this_block = true;
}

bool spip::IBVReceiver::process_packet (const std::uint8_t *data, std::size_t length)
{
  if (length < packet_size)
  {
    cerr << "Received packet of " << length << " bytes, expected " << packet_size << endl;
    return false;
  }

  if (need_next_block)
  {
    open_block();
  }

  // decode the header so that the format knows what to do with the packet
  byte_offset = format->decode_packet (data, &bytes_received);

  // if we do not yet have a UTC start, get it from the format
  if (!have_utc_start)
  {
    Time utc_start = format->get_utc_start ();
    uint64_t pico_seconds = format->get_pico_seconds();
    have_utc_start = true;
  }

  // ignore if negative
  if (byte_offset < 0)
  {
    // data stream is finished
    if (byte_offset == -2)
    {
      set_control_cmd(Stop);
    }
  }
  // packet is part of this observation
  else
  {
    // packet belongs in current buffer
    if ((byte_offset >= curr_byte_offset) && (byte_offset < next_byte_offset))
    {
      bytes_this_buf += bytes_received;
      stats->increment_bytes (bytes_received);
      format->insert_last_packet (block + (byte_offset - curr_byte_offset));
      queue->consume_packet();
    }
    // packet belongs in overflow buffer
    else if ((byte_offset >= next_byte_offset) && (byte_offset < overflow_maxbyte))
    {
      format->insert_last_packet (overflow + (byte_offset - next_byte_offset));
      overflow->copied_from (byte_offset - next_byte_offset, bytes_received);
      queue->consume_packet();
    } 
    // ignore packets that preceed this buffer
    else if (byte_offset < curr_byte_offset)
    {
      queue->consume_packet();
    }
    else
    {
      filled_this_block = true;
    }
  }

  // close open data block buffer if is is now full
  if (bytes_this_buf >= int64_t(data_block_bufsz) || filled_this_block)
  {
    close_block();
    stats->dropped_bytes (data_block_bufsz - bytes_this_buf);
  }
}

void spip::IBVReceiver::stop_receiving ()
{
  //keep_receiving = false;
  if (format)
    format->conclude();
}
