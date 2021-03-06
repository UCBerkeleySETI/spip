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
#include <boost/asio.hpp>

using namespace std;

spip::IBVReceiver::IBVReceiver()
{
  keep_receiving = true;
  have_utc_start = false;
  format = NULL;
  verbose = 1;

  boost::asio::io_service io;
  queue = new IBVQueue(io);
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

  // determine the resolution of the format 
  uint64_t resolution = format->get_resolution();
  size_t npackets = 16384;
  // configure the queue
  packet_size = format->get_packet_size();
  header_size = format->get_header_size();
  if (verbose)
    cerr << "spip::IBVReceiver::configure queue->configure()" << endl;
  queue->configure (npackets, packet_size, header_size);

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
  if (verbose)
    cerr << "spip::IBVReceiver::configure queue->allocate()" << endl;
  queue->allocate ();

  if (data_mcast.size() > 0)
  {
    if (verbose)
      cerr << "spip::IBVReceiver::configure queue->join_multicast()" << endl;
    queue->join_multicast (data_host, data_port);
  }
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
      utc_start.add_seconds (5);
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

  // virtual block, make about 128 MB
  data_bufsz = nchan * ndim * npol;
  while (data_bufsz < 128*1024*1024)
    data_bufsz *= 2;

  curr_block = (char *) malloc (data_bufsz);
  next_block = (char *) malloc (data_bufsz);

  bytes_curr_buf = 0;
  bytes_next_buf = 0;

  // block accounting 
  curr_byte_offset = 0;
  next_byte_offset = 0;
  last_byte_offset = 0;

#ifdef _DEBUG
  cerr << "spip::IBVReceiver::receive [" << curr_byte_offset << " - "
       << next_byte_offset << " - " << last_byte_offset << "]" << endl;
#endif

  int64_t byte_offset;
  unsigned bytes_received;

  if (verbose)
    cerr << "spip::IBVReceiver::receive starting main loop" << endl;

  spip::IBVQueue::keep_receiving = true;
  need_next_block = true;

  while (spip::IBVQueue::keep_receiving)
  {
    int got = queue->open_packet ();
    if (got > 0)
    {
      if (need_next_block)
      {
#ifdef _DEBUG
        cerr << "spip::IBVReceiver::receive open_block()" << endl;
#endif
        open_block();
      }

      // decode the header so that the format knows what to do with the packet
      byte_offset = format->decode_packet ((char *) queue->buf_ptr, &bytes_received);
  
      // if we do not yet have a UTC start, get it from the format
      if (!have_utc_start)
      { 
        Time utc_start = format->get_utc_start ();
        //uint64_t pico_seconds = format->get_pico_seconds();
        have_utc_start = true;
      }
  
      // ignore if negative
      if (byte_offset < 0)
      { 
        // data stream is finished
        if (byte_offset == -2)
        { 
          //set_control_cmd(Stop);
        }
        queue->close_packet();
      }
      // packet is part of this observation
      else
      { 
        // packet belongs in current buffer
        if ((byte_offset >= curr_byte_offset) && (byte_offset < next_byte_offset))
        { 
          bytes_curr_buf += bytes_received;
          stats->increment_bytes (bytes_received);
          format->insert_last_packet (curr_block + (byte_offset - curr_byte_offset));
          queue->close_packet();
        }
        // packet belongs in overflow buffer
        else if ((byte_offset >= next_byte_offset) && (byte_offset < last_byte_offset))
        {
          bytes_next_buf += bytes_received;
          format->insert_last_packet (next_block + (byte_offset - next_byte_offset));
          stats->increment_bytes (bytes_received);
          queue->close_packet();
        } 
        // ignore packets that preceed this buffer
        else if (byte_offset < curr_byte_offset)
        { 
          queue->close_packet();
        }
        else
        { 
          filled_this_block = true;
        }
      }
  
      // close open data block buffer if is is now full
      if (bytes_curr_buf >= data_bufsz || filled_this_block)
      {
#ifdef _DEBUG
        cerr << "spip::IBVReceiver::process_packet close_block() " << data_bufsz - bytes_curr_buf << endl;
#endif
        close_block();
        stats->dropped_bytes (data_bufsz - bytes_curr_buf);

        // update stats for any sleeps
        uint64_t nsleeps = queue->process_sleeps();
        stats->sleeps(nsleeps);

      }
    }
    else if (got == 0)
    {
#ifdef _TRACE
      cerr << "spip::IBVReceiver::receive no packets in queue" << endl;
#endif
    }
    else
    {
      cerr << "spip::IBVReceiver::receive queue->open_packet returned " << got << endl;
    }
  }
}

void spip::IBVReceiver::open_block ()
{
  need_next_block = false;

  // here we would open a ring buffer
  char * tmp = curr_block;
  curr_block = next_block;
  next_block = tmp;

  // update absolute limits
  curr_byte_offset = next_byte_offset;
  next_byte_offset = curr_byte_offset + data_bufsz;
  last_byte_offset = next_byte_offset + data_bufsz;

  // copy any data from the overflow into the new block
  bytes_curr_buf = bytes_next_buf;
  bytes_next_buf = 0;

#ifdef _DEBUG
  cerr << "spip::UDPReceiveDB::open_block filling buffer ["
       << curr_byte_offset << " - " << next_byte_offset
       << " - " << last_byte_offset << "] " << endl;
#endif

  filled_this_block = false;
}

void spip::IBVReceiver::close_block ()
{
  need_next_block = true;
}

void spip::IBVReceiver::stop_receiving ()
{
  spip::IBVQueue::keep_receiving = false;
  if (format)
    format->conclude();
}
