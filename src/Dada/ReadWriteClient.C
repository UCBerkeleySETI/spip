/***************************************************************************
 *
 *   Copyright (C) 2017 Andrew Jameson
 *   Licensed under the Academic Free License version 2.1
 *
 ***************************************************************************/

#include "spip/ReadWriteClient.h"

using namespace std;

spip::ReadWriteClient::ReadWriteClient (const char * read_key_string, const char * write_key_string)
{
  if (verbose)
    cerr << "spip::ReadWriteClient::ReadWriteClient()" << endl;
  read_db = new DataBlockRead (read_key_string);
  read_db->connect();
  read_db->lock();

  write_db = new DataBlockWrite (write_key_string);
  write_db->connect();
  write_db->lock();

  buffer_size = 0;
  read_buffer = NULL;
  write_buffer = NULL;
}

spip::ReadWriteClient::~ReadWriteClient ()
{
  if (verbose)
    cerr << "spip::ReadWriteClient::~ReadWriteClient()" << endl;
  read_db->unlock();
  read_db->disconnect();
  delete read_db;
  write_db->unlock();
  write_db->disconnect();
  delete write_db;

  if (read_buffer)
    free (read_buffer);
  read_buffer = 0;
  if (write_buffer)
    free (write_buffer);
  write_buffer = 0;
}

int64_t spip::ReadWriteClient::main ()
{
  // read the header from the data block, configuring AsciiHeader
  header.load_from_str (read_db->read_header());

  if (verbose)
    cerr << "spip::ReadWriteClient::main open()" << endl;
  // client's open function to inspect/modify the AsciiHeader
  open();

  // write the output header to the data block
  if (verbose)
    cerr << "spip::WriteClient::main db->write_header()" << endl;
  write_db->write_header (header.raw());

  // perform the data transfer
  if (verbose)
    cerr << "spip::ReadWriteClient::main db->io_loop()" << endl;
  io_loop();

  // call clients close method at the end of data
  if (verbose)
    cerr << "spip::ReadWriteClient::main close()" << endl;

  // return the total number of bytes read 
  return bytes_transferred_loop;
}

void spip::ReadWriteClient::prepare()
{
  if (verbose)
    cerr << "spip::ReadWriteClient::prepare()" << endl;

  // if the buffers are not large enough
  if (int64_t(buffer_size) != optimal_bytes)
  {
    int rval;
    buffer_size = optimal_bytes;

    rval = posix_memalign ( (void **) &read_buffer, 512, optimal_bytes);
    if (rval != 0)
      cerr << "spip::ReadWriteClient::prepare posix_memalign failed" << endl;

    rval = posix_memalign ( (void **) &write_buffer, 512, optimal_bytes);
    if (rval != 0)
      cerr << "spip::ReadWriteClient::prepare posix_memalign failed" << endl;
  }
}

// transfer data to the data block until end condition is met
int64_t spip::ReadWriteClient::io_loop ()
{
  prepare ();

  int64_t bytes_remaining, bytes_to_transfer, bytes_read, bytes_written;
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
      cerr << "spip::ReadWriteClient::io_loop requesting " << bytes_to_transfer << " bytes from DB" << endl;

    // read from the data block into the interim buffer
    bytes_read = read_db->read (read_buffer, bytes_to_transfer);

    if (verbose)
      cerr << "spip::ReadWriteClient::io_loop read " << bytes_read << " bytes from DB" << endl;

    // check for end of data
    if (bytes_read <= 0)
    {
      if (verbose)
        cerr << "spip::ReadWriteClient::io_loop eod true" << endl;
      eod = true;
    }
    else
    {
      // call the client's io method
      bytes_written = io_data (read_buffer, write_buffer, bytes_read);
      if (bytes_written != bytes_read)
        throw Error (FailedCall, "spip::ReadWriteClient::io_loop", "io_data wrote fewer bytes than requested");

      // write the modified data to the output data block
      write_db->write_data (write_buffer, bytes_written);
    
      // update the number of bytes read
      bytes_transferred_loop += bytes_written;
    }
  }

  if (verbose)
    cerr << "spip::ReadWriteClient::io_loop bytes_transferred_loop=" << bytes_transferred_loop << endl;

  return bytes_transferred_loop;
}
