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
#include "spip/UDPReceiver.h"
#include "sys/time.h"

#ifdef HAVE_VMA
#include "spip/UDPSocketReceiveVMA.h"
#else
#include "spip/UDPSocketReceive.h"
#endif

#include <unistd.h>
#include <cstring>
#include <iostream>
#include <stdexcept>
#include <new>

using namespace std;

spip::UDPReceiver::UDPReceiver()
{
  keep_receiving = true;
  have_utc_start = false;
  format = NULL;
  verbose = 1;
}

spip::UDPReceiver::~UDPReceiver()
{
  if (format)
  {
    format->conclude();
    delete format;
  }
}

void spip::UDPReceiver::configure (const char * config_str)
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
    cerr << "spip::UDPReceiver::configure receiving on " 
         << data_host << ":" << data_port  << endl;

  uint64_t mega = 1e6;
  bits_per_second = (mega * nchan * npol * ndim * nbit) / tsamp;
  bytes_per_second = bits_per_second / 8;

  if (!format)
    throw runtime_error ("format was not allocated");
  format->configure (header, "");
}

void spip::UDPReceiver::prepare ()
{
  if (verbose)
    cerr << "spip::UDPReceiver::prepare()" << endl;

  stats = new UDPStats (format->get_header_size(), format->get_data_size());

  // if this format is not self starting, check for the UTC_START
  if (!format->get_self_start ())
  {
    // check if UTC_START has been set
    char * buffer = (char *) malloc (128);
    if (header.get ("UTC_START", "%s", buffer) == -1)
    {
      cerr << "spip::UDPReceiver::open no UTC_START in header" << endl;
      time_t now = time(0);
      spip::Time utc_start (now);
      utc_start.add_seconds (2);
      std::string utc_str = utc_start.get_gmtime();
      cerr << "spip::UDPReceiver::open UTC_START=" << utc_str  << endl;
      if (header.set ("UTC_START", "%s", utc_str.c_str()) < 0)
        throw invalid_argument ("failed to write UTC_START to header");
    }
    have_utc_start = true;
  }

  format->prepare (header, "");
}

void spip::UDPReceiver::set_format (spip::UDPFormat * fmt)
{
  if (format)
    delete format;
  format = fmt;
}

// receive UDP packets for the specified time at the specified data rate
void spip::UDPReceiver::receive ()
{
  if (verbose)
    cerr << "spip::UDPReceiver::receive()" << endl;

  ssize_t got;
  uint64_t nsleeps = 0;

  if (verbose)
    cerr << "spip::UDPReceiver::receive()" << endl;

  // create and open a UDP receiving socket
#ifdef HAVE_VMA
  UDPSocketReceiveVMA * sock = new UDPSocketReceiveVMA ();
#else
  UDPSocketReceive * sock = new UDPSocketReceive ();
#endif

  if (data_mcast.size() > 0)
  {
    cerr << "spip::UDPReceiver::receive sock->open_multicast (" << data_host << ", " << data_mcast << ", " << data_port << ")" << endl;
    sock->open_multicast (data_host, data_mcast, data_port);
  }
  else
  {
    if (verbose)
      cerr << "spip::UDPReceiver::receive sock->open(" << data_host << ", " << data_port << ")" << endl;
    sock->open (data_host, data_port);
  }

  // expected size of a UDP packet
  int packet_size = int(format->get_packet_size());
  size_t sock_size = packet_size + 64;
  if (verbose)
    cerr << "spip::UDPReceiver::receive sock->resize(" << sock_size << ")" << endl;
  sock->resize (sock_size);
  sock->resize_kernel_buffer (64*1024*1024);

  // virtual block, make about 128 MB
  size_t data_bufsz = nchan * ndim * npol;
  while (data_bufsz < 128*1024*1024)
    data_bufsz *= 2;

  char * curr_block = (char *) malloc (data_bufsz);
  char * next_block = (char *) malloc (data_bufsz);
  bool need_next_block = false;

  // block accounting 
  int64_t curr_byte_offset = 0;
  int64_t next_byte_offset = curr_byte_offset + data_bufsz;
  int64_t last_byte_offset = curr_byte_offset + data_bufsz;

#ifdef _DEBUG
  cerr << "spip::UDPReceiver::receive [" << curr_byte_offset << " - "
       << next_byte_offset << "] (" << 0 << ")" << endl;
#endif

  int64_t bytes_curr_buf = 0;
  int64_t bytes_next_buf = 0;
  int64_t byte_offset;
  unsigned bytes_received;

  if (verbose)
    cerr << "spip::UDPReceiver::receive starting main loop" << endl;

  // clear all packets buffered at the socket
  sock->clear_buffered_packets();

  while (sock->still_receiving())
  {
    // get a packet from the socket
    got = sock->recv_from();

    // received a bad packet, brutal exit
    if (got < packet_size)
    {
      sock->consume_packet();
      //cerr << "Received UDP packet of " << got << " bytes, expected " << packet_size << endl;
    }
    else
    {
      if (need_next_block)
      {
        need_next_block = false;

        // update absolute limits
        curr_byte_offset = next_byte_offset;
        next_byte_offset = curr_byte_offset + data_bufsz;
        last_byte_offset = next_byte_offset + data_bufsz;

        bytes_curr_buf = bytes_next_buf;
        bytes_next_buf = 0;

#ifdef _DEBUG
        cerr << "spip::UDPReceiver::receive [" << curr_byte_offset << " - "
             << next_byte_offset << " - " << last_byte_offset 
             << "] (" << bytes_curr_buf << ")" << endl;
#endif
        // update stats for any sleeps
        nsleeps = sock->process_sleeps();
        stats->sleeps(nsleeps);
      }

      // decode the header so that the format knows what to do with the packet
      byte_offset = format->decode_packet (sock->buf_ptr, &bytes_received);

      // if we do not yet have a UTC start, get it from the format
      if (!have_utc_start)
      {
        Time utc_start = format->get_utc_start ();
        uint64_t pico_seconds = format->get_pico_seconds();
        have_utc_start = true;
      }

      // packet belongs in current buffer
      if ((byte_offset >= curr_byte_offset) && (byte_offset < next_byte_offset))
      {
        stats->increment_bytes (bytes_received);
        format->insert_last_packet (curr_block + (byte_offset - curr_byte_offset));
        bytes_curr_buf += bytes_received;
        sock->consume_packet();
      }
      else if ((byte_offset >= next_byte_offset) && (byte_offset < last_byte_offset))
      {
        stats->increment_bytes (bytes_received);
        format->insert_last_packet (next_block + (byte_offset - next_byte_offset));
        bytes_next_buf += bytes_received;
        sock->consume_packet();
      }
      // ignore
      else if (byte_offset < curr_byte_offset)
      {
        sock->consume_packet();
      }
      else
      {
#ifdef _DEBUG
        cerr << "ELSE byte_offset=" << byte_offset << " [" << curr_byte_offset <<" - " << next_byte_offset << " - " << last_byte_offset << "] bytes_received=" << bytes_received << " bytes_curr_buf=" << bytes_curr_buf << endl; 
#endif
        need_next_block = true;
      }

      if (bytes_curr_buf >= data_bufsz || bytes_next_buf >= data_bufsz/2 || need_next_block)
      {
        stats->dropped_bytes (data_bufsz - bytes_curr_buf);
        need_next_block = true;
      }
    }
  }
  sock->close_me();
}

void spip::UDPReceiver::stop_receiving ()
{
  //keep_receiving = false;
  if (format)
    format->conclude();
}
