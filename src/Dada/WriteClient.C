/***************************************************************************
 *
 *   Copyright (C) 2017 Andrew Jameson
 *   Licensed under the Academic Free License version 2.1
 *
 ***************************************************************************/

#include "spip/WriteClient.h"

using namespace std;

spip::WriteClient::WriteClient (const char * key_string)
{
  if (verbose)
    cerr << "spip::WriteClient::WriteClient()" << endl;
  db = new DataBlockWrite (key_string);
  db->connect();
  db->lock();
}

spip::WriteClient::~WriteClient ()
{
  if (verbose)
    cerr << "spip::WriteClient::~WriteClient()" << endl;
  db->unlock();
  db->disconnect();
  delete db;
}

int64_t spip::WriteClient::main ()
{
  if (verbose)
    cerr << "spip::WriteClient::main open()" << endl;
  // write client's open function to populate the Ascii header
  open();

  // write the header to the data block
  if (verbose)
    cerr << "spip::WriteClient::main db->write_header()" << endl;
  db->write_header (header.raw());

  // perform the data transfer
  if (verbose)
    cerr << "spip::WriteClient::main db->io_loop()" << endl;
  io_loop();

  // call clients close method at the end of data
  if (verbose)
    cerr << "spip::WriteClient::main close()" << endl;

  // return the total number of bytes_to_write transferred
  return bytes_written_loop;
}

// transfer data to the data block until end condition is met
int64_t spip::WriteClient::io_loop ()
{
  char * buffer = 0;
  size_t buffer_size = 0;

  if (int64_t(buffer_size) != optimal_bytes)
  {
    buffer_size = optimal_bytes;
    int rval = posix_memalign ( (void **) &buffer, 512, optimal_bytes);
    if (rval != 0)
      cerr << "spip::WriteClient::io_loop posix_memalign failed" << endl;
  }

  int64_t bytes_remaining, bytes_to_write, bytes_written;
  bytes_written_loop = 0;
  while (!transfer_bytes || bytes_written_loop < transfer_bytes)
  {
    // determine how many bytes_to_write to transfer on this io_loop
    if (!transfer_bytes)
    {
      bytes_to_write = buffer_size;
    }
    else
    {  
      bytes_remaining = transfer_bytes - bytes_written_loop;
      if (int64_t(buffer_size) > bytes_remaining)
        bytes_to_write = bytes_remaining;
      else
        bytes_to_write = buffer_size;
    }

    // call the WriteClient's io method
    bytes_written = io (buffer, bytes_to_write);
    if (bytes_written != bytes_to_write)
      throw Error (FailedCall, "spip::WriteClient::io_loop", "io wrote fewer bytes than requested");

    // use the flexible write_data method
    db->write_data (buffer, bytes_written);

    bytes_written_loop += bytes_written;
  }

  return bytes_written_loop;
}
