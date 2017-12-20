/***************************************************************************
 *
 *   Copyright (C) 2017 Andrew Jameson
 *   Licensed under the Academic Free License version 2.1
 *
 ***************************************************************************/

#include "spip/ReadReadWriteBlockClient.h"

using namespace std;

spip::ReadReadWriteBlockClient::ReadReadWriteBlockClient (const char * read1_key_string, 
                                                const char * read2_key_string,
                                                const char * write_key_string)
{
  if (verbose)
    cerr << "spip::ReadReadWriteBlockClient::ReadReadWriteBlockClient()" << endl;

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

spip::ReadReadWriteBlockClient::~ReadReadWriteBlockClient ()
{
  if (verbose)
    cerr << "spip::ReadReadWriteBlockClient::~ReadReadWriteBlockClient()" << endl;

  read1_db->unlock();
  read1_db->disconnect();
  delete read1_db;

  read2_db->unlock();
  read2_db->disconnect();
  delete read2_db;

  write_db->unlock();
  write_db->disconnect();
  delete write_db;
}

int64_t spip::ReadReadWriteBlockClient::main ()
{
  if (verbose)
    cerr << "spip::ReadReadWriteBlockClient::main configure()" << endl;
  // some basic checks
  configure();

  if (verbose)
    cerr << "spip::ReadReadWriteBlockClient::main read1_header.load_from_str()" << endl;
  // read the header from the 1st data block, configuring AsciiHeader
  read1_header.load_from_str (read1_db->read_header());

  if (verbose)
    cerr << "spip::ReadReadWriteBlockClient::main read2_header.load_from_str()" << endl;
  // read the header from the 2nddata block, configuring AsciiHeader
  read2_header.load_from_str (read2_db->read_header());

  if (verbose)
    cerr << "spip::ReadReadWriteBlockClient::main open()" << endl;
  // client's open function to inspect/modify the AsciiHeaders
  open();

  // write the output header to the data block
  if (verbose)
    cerr << "spip::WriteClient::main db->write_header()" << endl;
  write_db->open ();
  write_db->write_header (write_header.raw());

  // perform the data transfer
  if (verbose)
    cerr << "spip::ReadReadWriteBlockClient::main db->io_loop()" << endl;
  io_loop();

  // call clients close method at the end of data
  if (verbose)
    cerr << "spip::ReadReadWriteBlockClient::main close()" << endl;

  // close the output data block
  write_db->close();

  // return the total number of bytes read 
  return bytes_transferred_loop;
}

void spip::ReadReadWriteBlockClient::configure ()
{
  read1_block_size = read1_db->get_data_bufsz();
  read2_block_size = read2_db->get_data_bufsz();
  write_block_size = write_db->get_data_bufsz();

  // TODO we can generalise this later
  if (read1_block_size != read2_block_size)
    throw Error (InvalidState, "spip::ReadReadWriteBlockClient::configure",
                 "size of read1 and read2 data block buffers were different");

  //if (read1_block_size + read2_block_size != write_block_size)
  //  throw Error (InvalidState, "spip::ReadReadWriteBlockClient::configure",
  //               "sizeof (read1 + read2) != sizeof (write) data blocks");
}

// transfer data to the data block until end condition is met
int64_t spip::ReadReadWriteBlockClient::io_loop ()
{
  int64_t bytes_remaining, bytes_to_transfer, bytes_written;
  bytes_transferred_loop = 0;
  bool eod = false;
  uint64_t read1_bytes, read2_bytes;

  if (verbose)
    cerr << "spip::ReadReadWriteBlockClient::io_loop transfer_bytes=" << transfer_bytes << endl;

  while (!eod && (!transfer_bytes || bytes_transferred_loop < transfer_bytes))
  {
    // determine how many bytes to transfer on this io_loop
    if (!transfer_bytes)
    {
      bytes_to_transfer = read1_block_size;
    }
    else
    {
      bytes_remaining = transfer_bytes - bytes_transferred_loop;
      if (int64_t(read1_block_size) > bytes_remaining)
        bytes_to_transfer = bytes_remaining;
      else
        bytes_to_transfer = int64_t(read1_block_size);
    }
    if (verbose)
      cerr << "spip::ReadReadWriteBlockClient::io_loop requesting " << bytes_to_transfer << " bytes from DB" << endl;

    // open blocks from the read and write DBs
    read1_buffer = (void *) read1_db->open_block();
    read2_buffer = (void *) read2_db->open_block();

    read1_bytes = read1_db->get_buf_bytes();
    read2_bytes = read2_db->get_buf_bytes();

    // check for end of data
    if (read1_buffer == NULL || read2_buffer == NULL)
    {
      if (verbose)
        cerr << "spip::ReadReadWriteBlockClient::io_loop eod true" << endl;
      eod = true;
    }
    else
    {
      write_buffer = (void *) write_db->open_block();

      // call the client's io method
      bytes_written = io_block(read1_buffer, read2_buffer, write_buffer, write_block_size);

      if (bytes_written != int64_t(write_block_size))
        throw Error (FailedCall, "spip::ReadReadWriteBlockClient::io_loop", "io_data wrote fewer bytes than requested");

      // write the modified data to the output data block
      write_db->close_block (bytes_written);
    
      // update the number of bytes read
      bytes_transferred_loop += bytes_written;

      read1_db->close_block(read1_bytes);
      read2_db->close_block(read2_bytes);
    }
  }

  if (verbose)
    cerr << "spip::ReadReadWriteBlockClient::io_loop bytes_transferred_loop=" << bytes_transferred_loop << endl;

  return bytes_transferred_loop;
}
