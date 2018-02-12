/***************************************************************************
 *
 *   Copyright (C) 2017 Andrew Jameson
 *   Licensed under the Academic Free License version 2.1
 *
 ***************************************************************************/

#include "spip/ReadWriteBlockClient.h"

using namespace std;

spip::ReadWriteBlockClient::ReadWriteBlockClient (const char * read_key_string, 
                                                          const char * write_key_string)
{
  if (verbose)
    cerr << "spip::ReadWriteBlockClient::ReadWriteBlockClient()" << endl;

  read_db = new DataBlockRead (read_key_string);
  read_db->connect();
  read_db->lock();

  write_db = new DataBlockWrite (write_key_string);
  write_db->connect();
  write_db->lock();

  read_buffer_size = 0;
  write_buffer_size = 0;
  read_buffer = NULL;
  write_buffer = NULL;
}

spip::ReadWriteBlockClient::~ReadWriteBlockClient ()
{
  if (verbose)
    cerr << "spip::ReadWriteBlockClient::~ReadWriteBlockClient()" << endl;

  read_db->unlock();
  read_db->disconnect();
  delete read_db;

  write_db->unlock();
  write_db->disconnect();
  delete write_db;
}

int64_t spip::ReadWriteBlockClient::main ()
{
  if (verbose)
    cerr << "spip::ReadWriteBlockClient::main configure()" << endl;
  // some basic checks
  configure();

  if (verbose)
    cerr << "spip::ReadWriteBlockClient::main read_header.load_from_str()" << endl;
  // read the header from the input data block, configuring AsciiHeader
  read_header.load_from_str (read_db->read_header());

  if (verbose)
    cerr << "spip::ReadWriteBlockClient::main open()" << endl;
  // client's open function to inspect/modify the AsciiHeaders
  open();

  // write the output header to the data block
  if (verbose)
    cerr << "spip::WriteClient::main db->write_header()" << endl;
  write_db->open ();
  write_db->write_header (write_header.raw());

  // perform the data transfer
  if (verbose)
    cerr << "spip::ReadWriteBlockClient::main db->io_loop()" << endl;
  io_loop();

  // call clients close method at the end of data
  if (verbose)
    cerr << "spip::ReadWriteBlockClient::main close()" << endl;

  // close the output data block
  write_db->close();

  // return the total number of bytes read 
  return bytes_transferred_loop;
}

void spip::ReadWriteBlockClient::configure ()
{
  read_block_size = read_db->get_data_bufsz();
  write_block_size = write_db->get_data_bufsz();
}

// transfer data to the data block until end condition is met
int64_t spip::ReadWriteBlockClient::io_loop ()
{
  int64_t bytes_remaining, bytes_to_transfer, bytes_written;
  bytes_transferred_loop = 0;
  bool eod = false;
  uint64_t read_bytes;

  if (verbose)
    cerr << "spip::ReadWriteBlockClient::io_loop transfer_bytes=" << transfer_bytes << endl;

  while (!eod && (!transfer_bytes || bytes_transferred_loop < transfer_bytes))
  {
    // determine how many bytes to transfer on this io_loop
    if (!transfer_bytes)
    {
      bytes_to_transfer = read_block_size;
    }
    else
    {
      bytes_remaining = transfer_bytes - bytes_transferred_loop;
      if (int64_t(read_block_size) > bytes_remaining)
        bytes_to_transfer = bytes_remaining;
      else
        bytes_to_transfer = int64_t(read_block_size);
    }
    if (verbose)
      cerr << "spip::ReadWriteBlockClient::io_loop requesting " << bytes_to_transfer << " bytes from DB" << endl;

    // open blocks from the read and write DBs
    read_buffer = (void *) read_db->open_block();
    read_bytes = read_db->get_buf_bytes();

    // check for end of data
    if (read_buffer == NULL)
    {
      if (verbose)
        cerr << "spip::ReadWriteBlockClient::io_loop eod true" << endl;
      eod = true;
    }
    else
    {
      write_buffer = (void *) write_db->open_block();

      // call the client's io method
      bytes_written = io_block(read_buffer, write_buffer, read_block_size);

      if (bytes_written != int64_t(write_block_size))
        throw Error (FailedCall, "spip::ReadWriteBlockClient::io_loop", "io_data wrote fewer bytes than requested");

      // write the modified data to the output data block
      write_db->close_block (bytes_written);
    
      // update the number of bytes read
      bytes_transferred_loop += bytes_written;

      read_db->close_block(read_bytes);
    }
  }

  if (verbose)
    cerr << "spip::ReadWriteBlockClient::io_loop bytes_transferred_loop=" << bytes_transferred_loop << endl;

  return bytes_transferred_loop;
}
