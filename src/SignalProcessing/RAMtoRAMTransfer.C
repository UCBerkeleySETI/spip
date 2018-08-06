/***************************************************************************
 *
 *   Copyright (C) 2016 Andrew Jameson
 *   Licensed under the Academic Free License version 2.1
 *
 ***************************************************************************/

#include "spip/RAMtoRAMTransfer.h"

#include <stdexcept>
#include <string.h>

using namespace std;

spip::RAMtoRAMTransfer::RAMtoRAMTransfer () : Transformation<Container,Container>("RAMtoRAMTransfer", outofplace)
{
  nblock_out = 1;
  iblock_out = 0;
}

spip::RAMtoRAMTransfer::~RAMtoRAMTransfer ()
{
}

void spip::RAMtoRAMTransfer::set_output_reblock (unsigned factor)
{
  nblock_out = factor;
}

//! intial configuration at the start of the data stream
void spip::RAMtoRAMTransfer::configure (spip::Ordering output_order)
{
  ndat  = input->get_ndat ();

  if (verbose)
    cerr << "spip::RAMtoRAMTransfer::configure: output->clone_header" << endl;

  // copy input header to output
  output->clone_header (input->get_header());

  // output will read the newly cloned header parameters
  output->read_header ();

  // update the output header parameters with the new details
  output->write_header ();
  
  // resize the output container
  prepare_output();

  // 
  iblock_out = 0;
}

void spip::RAMtoRAMTransfer::prepare ()
{
  ndat  = input->get_ndat ();
  if (verbose)
    cerr << "spip::RAMtoRAMTransfer::prepare ndat=" << ndat << endl;
}

//! simply copy input buffer to output buffer
void spip::RAMtoRAMTransfer::transformation ()
{
  if (verbose)
    cerr << "spip::RAMtoRAMTransfer::transformation()" << endl;

  // ensure output is appropriately sized
  prepare_output ();

  uint64_t block_stride = input->get_size();

  void * in = (void *) input->get_buffer();
  void * out = (void *) (output->get_buffer() + (iblock_out * block_stride));
  size_t nbytes = input->calculate_buffer_size();

  if (verbose)
    cerr << "spip::RAMtoRAMTransfer::transformation memcpy (" << (void *) out << ", "
         << (void *) in << ", " << nbytes << ")" << endl;

  // perform host to device transfer TODO check for smaller buffesr
  memcpy (out, in, nbytes);

  // increment the output reblocking factor
  iblock_out++;

  // reset once the ouput is full
  if (iblock_out == nblock_out)
  {
    iblock_out = 0;
  }
}

void spip::RAMtoRAMTransfer::prepare_output ()
{
  output->set_ndat (ndat * nblock_out);
  output->resize();
}

