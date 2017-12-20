/***************************************************************************
 *
 *   Copyright (C) 2017 Andrew Jameson
 *   Licensed under the Academic Free License version 2.1
 *
 ***************************************************************************/

#include "spip/ReadReadWriteClient.h"

using namespace std;

spip::ReadReadWriteClient::ReadReadWriteClient (const char * read1_key_string, 
                                                const char * read2_key_string,
                                                const char * write_key_string)
{
  if (verbose)
    cerr << "spip::ReadReadWriteClient::ReadReadWriteClient()" << endl;

  read1_db = new DataBlockRead (read1_key_string);
  read1_db->connect();
  read1_db->lock();

  read2_db = new DataBlockRead (read2_key_string);
  read2_db->connect();
  read2_db->lock();

  write_db = new DataBlockWrite (write_key_string);
  write_db->connect();
  write_db->lock();

  read_buffer_size = 0;
  write_buffer_size = 0;
  read1_buffer = NULL;
  read2_buffer = NULL;
  write_buffer = NULL;
}

spip::ReadReadWriteClient::~ReadReadWriteClient ()
{
  if (verbose)
    cerr << "spip::ReadReadWriteClient::~ReadReadWriteClient()" << endl;

  read1_db->unlock();
  read1_db->disconnect();
  delete read1_db;

  read2_db->unlock();
  read2_db->disconnect();
  delete read2_db;

  write_db->unlock();
  write_db->disconnect();
  delete write_db;

  if (read1_buffer)
    free (read1_buffer);
  read1_buffer = 0;
  if (read2_buffer)
    free (read2_buffer);
  read2_buffer = 0;
  if (write_buffer)
    free (write_buffer);
  write_buffer = 0;
}

int64_t spip::ReadReadWriteClient::main ()
{
  // read the header from the 1st data block, configuring AsciiHeader
  read1_header.load_from_str (read1_db->read_header());

  // read the header from the 2nddata block, configuring AsciiHeader
  read2_header.load_from_str (read2_db->read_header());

  if (verbose)
    cerr << "spip::ReadReadWriteClient::main open()" << endl;
  // client's open function to inspect/modify the AsciiHeaders
  open();

  // write the output header to the data block
  if (verbose)
    cerr << "spip::WriteClient::main db->write_header()" << endl;
  write_db->write_header (write_header.raw());

  // perform the data transfer
  if (verbose)
    cerr << "spip::ReadReadWriteClient::main db->io_loop()" << endl;
  io_loop();

  // call clients close method at the end of data
  if (verbose)
    cerr << "spip::ReadReadWriteClient::main close()" << endl;

  // return the total number of bytes read 
  return bytes_transferred_loop;
}

void spip::ReadReadWriteClient::prepare()
{
  if (verbose)
    cerr << "spip::ReadReadWriteClient::prepare()" << endl;

  // if the buffers are not large enough
  if (int64_t(buffer_size) != optimal_bytes)
  {
    int rval;
    buffer_size = optimal_bytes;

    rval = posix_memalign ( (void **) &read1_buffer, 512, optimal_bytes);
    if (rval != 0)
      cerr << "spip::ReadReadWriteClient::prepare posix_memalign failed" << endl;

    rval = posix_memalign ( (void **) &read2_buffer, 512, optimal_bytes);
    if (rval != 0)
      cerr << "spip::ReadReadWriteClient::prepare posix_memalign failed" << endl;

    rval = posix_memalign ( (void **) &write_buffer, 512, (optimal_bytes*2));
    if (rval != 0)
      cerr << "spip::ReadReadWriteClient::prepare posix_memalign failed" << endl;
  }
}

// transfer data to the data block until end condition is met
int64_t spip::ReadReadWriteClient::io_loop ()
{
  prepare ();

  int64_t bytes_remaining, bytes_to_transfer, bytes_read1;
  int64_t bytes_read2, bytes_to_io, bytes_written;
  bytes_transferred_loop = 0;
  bool eod = false;

  while (!eod && (!transfer_bytes || bytes_transferred_loop < transfer_bytes))
  {
    // determine how many bytes to transfer on this io_loop
    if (!transfer_bytes)
    {
      bytes_to_transfer = buffer_size;
    }
    else
    {
      bytes_remaining = transfer_bytes - bytes_transferred_loop;
      if (int64_t(buffer_size) > bytes_remaining)
        bytes_to_transfer = bytes_remaining;
      else
        bytes_to_transfer = int64_t(buffer_size);
    }
    if (verbose)
      cerr << "spip::ReadReadWriteClient::io_loop requesting " << bytes_to_transfer << " bytes from DB" << endl;

    // read from the data blocks into the interim buffers
    bytes_read1 = read1_db->read (read1_buffer, bytes_to_transfer);
    bytes_read2 = read2_db->read (read2_buffer, bytes_to_transfer);

    if (verbose)
      cerr << "spip::ReadReadWriteClient::io_loop read " << bytes_read1 << " and "
           << bytes_read2 << " bytes from DBs" << endl;

    // check for unmatched reads
    if (bytes_read1 != bytes_read2)
      throw Error (InvalidState, "spip::ReadReadWriteClient::io_loop", 
                   "read different number of bytes from each DB");

    // check for end of data
    if (bytes_read1 <= 0 || bytes_read2 <= 0)
    {
      if (verbose)
        cerr << "spip::ReadReadWriteClient::io_loop eod true" << endl;
      eod = true;
    }
    else
    {
      bytes_to_io = bytes_read1 + bytes_read2;

      // call the client's io method
      bytes_written = io_data (read1_buffer, read2_buffer, write_buffer, bytes_to_io);
      if (bytes_written != bytes_to_io)
        throw Error (FailedCall, "spip::ReadReadWriteClient::io_loop", "io_data wrote fewer bytes than requested");

      // write the modified data to the output data block
      write_db->write_data (write_buffer, bytes_written);
    
      // update the number of bytes read
      bytes_transferred_loop += bytes_written;
    }
  }

  if (verbose)
    cerr << "spip::ReadReadWriteClient::io_loop bytes_transferred_loop=" << bytes_transferred_loop << endl;

  return bytes_transferred_loop;
}
