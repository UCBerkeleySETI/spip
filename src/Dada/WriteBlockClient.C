/***************************************************************************
 *
 *   Copyright (C) 2017 Andrew Jameson
 *   Licensed under the Academic Free License version 2.1
 *
 ***************************************************************************/

#include "spip/WriteBlockClient.h"

using namespace std;

spip::WriteBlockClient::WriteBlockClient (const char * key_string)
{
  db = new DataBlockWrite (key_string);
  db->connect();
  db->lock();
}

spip::WriteBlockClient::~WriteBlockClient ()
{
  db->unlock();
  db->disconnect();
  delete db;
}

int64_t spip::WriteBlockClient::main ()
{
  return 0;
}

// transfer data to the data block until end condition is met
int64_t spip::WriteBlockClient::io_loop ()
{
  // number of bytes_to_write to write
  int64_t bytes_written_loop = 0;    // bytes_to_write written by io_loop
  const uint64_t block_size = db->get_data_bufsz();
  void * block_buffer = 0;

  int64_t bytes_to_write, bytes_remaining, bytes_written;

  while (!transfer_bytes || bytes_written_loop < transfer_bytes)
  {
    // determine how many bytes_to_write to transfer on this io_loop
    if (!transfer_bytes)
    {
      bytes_to_write = block_size;
    }
    else
    {  
      bytes_remaining = transfer_bytes - bytes_written_loop;
      if (int64_t(block_size) > bytes_remaining)
        bytes_to_write = bytes_remaining;
      else
        bytes_to_write = block_size;
    }

    // open an empty data block buffer
    block_buffer = (void *) db->open_block();

    // subclass implements the io_block method for data write
    bytes_written = io_block (block_buffer, bytes_to_write);

    // error in io_block function, TODO  warning? exception?
    if (bytes_written < 0)
    {
    }
 
    // end of input from writing class
    if (bytes_written == 0)
      if (verbose)
        cerr << "spip::WriteBlockClient::io_loop io_block write 0 bytes, end of input" << endl; 

    // if the io_block resulted in the correct amount of data being written
    if (bytes_written_loop + bytes_written == transfer_bytes) 
    {
      if (db->update_block (bytes_written) < 0)
      {
        // TODO throw error update_block
      }
    }
    // 
    else
    {
      if (db->close_block (bytes_written) < 0)
      {
        // TODO throw error close_block
      }
    }
    
    if (bytes_written < int64_t(block_size)) 
    {
      
    }
     
    bytes_written_loop += bytes_written;
  }
  return bytes_written_loop;
}
