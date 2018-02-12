/***************************************************************************
 *
 *   Copyright (C) 2017 Andrew Jameson
 *   Licensed under the Academic Free License version 2.1
 *
 ***************************************************************************/

#include "spip/ReadClient.h"

using namespace std;

spip::ReadClient::ReadClient (const char * key_string)
{
  if (verbose)
    cerr << "spip::ReadClient::ReadClient()" << endl;
  db = new DataBlockRead (key_string);
  db->connect();
  db->lock();
}

spip::ReadClient::~ReadClient ()
{
  if (verbose)
    cerr << "spip::ReadClient::~ReadClient()" << endl;
  db->unlock();
  db->disconnect();
  delete db;
}

int64_t spip::ReadClient::main ()
{
  // read the header from the data block, configuring AsciiHeader
  header.load_from_str (db->read_header());

  if (verbose)
    cerr << "spip::ReadClient::main open()" << endl;
  // client's open function to inspect the AsciiHeader
  open();

  // perform the data transfer
  if (verbose)
    cerr << "spip::ReadClient::main db->io_loop()" << endl;
  io_loop();

  // call clients close method at the end of data
  if (verbose)
    cerr << "spip::ReadClient::main close()" << endl;

  // return the total number of bytes read 
  return bytes_read_loop;
}

// transfer data to the data block until end condition is met
int64_t spip::ReadClient::io_loop ()
{
  char * buffer = 0;
  size_t buffer_size = 0;

  if (int64_t(buffer_size) != optimal_bytes)
  {
    buffer_size = optimal_bytes;
    int rval = posix_memalign ( (void **) &buffer, 512, optimal_bytes);
    if (rval != 0)
      cerr << "spip::ReadClient::io_loop posix_memalign failed" << endl;
  }

  int64_t bytes_remaining, bytes_to_read, bytes_read;
  bytes_read_loop = 0;
  bool eod = false;
  while (!eod && (!transfer_bytes || bytes_read_loop < transfer_bytes))
  {
    // determine how many bytes to transfer on this io_loop
    if (!transfer_bytes)
    {
      bytes_to_read = buffer_size;
    }
    else
    {
      bytes_remaining = transfer_bytes - bytes_read_loop;
      if (int64_t(buffer_size) > bytes_remaining)
        bytes_to_read = bytes_remaining;
      else
        bytes_to_read = int64_t(buffer_size);
    }
    if (verbose)
      cerr << "spip::ReadClient::io_loop requesting " << bytes_to_read << " bytes from DB" << endl;

    // read from the data block into the interim buffer
    bytes_read = db->read (buffer, bytes_to_read);

    if (verbose)
      cerr << "spip::ReadClient::io_loop read " << bytes_read << " bytes from DB" << endl;

    // check for end of data
    if (bytes_read <= 0)
    {
      if (verbose)
        cerr << "spip::ReadClient::io_loop eod true" << endl;
      eod = true;
    }
    else
    {
      // call the client's io method
      io (buffer, bytes_read);
    
      // update the number of bytes read
      bytes_read_loop += bytes_read;
    }
  }

  if (verbose)
    cerr << "spip::ReadClient::io_loop bytes_read_loop=" << bytes_read_loop << endl;

  return bytes_read_loop;
}
