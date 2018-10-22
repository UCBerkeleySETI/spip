/***************************************************************************
 *
 *   Copyright (C) 2016 Andrew Jameson
 *   Licensed under the Academic Free License version 2.1
 *
 ***************************************************************************/

#include "config.h"

#include "spip/ContainerRAM.h"
#include "spip/Error.h"

#include <cstring>
#include <iostream>

#if HAVE_CUDA
#include <cuda_runtime.h>
#endif

using namespace std;

spip::ContainerRAM::ContainerRAM ()
{
  buffer_registered = false;
  buffer_should_register = false;
}

spip::ContainerRAM::~ContainerRAM ()
{
  free_buffer();
}

void spip::ContainerRAM::free_buffer()
{
  if (buffer)
  {
    if (buffer_registered)
      unregister_buffer();
    free (buffer);
    buffer = NULL;
  }
  size = 0;
}

void spip::ContainerRAM::resize ()
{
  if (spip::Container::verbose)
    cerr << "spip::ContainerRAM::resize()" << endl;

  uint64_t required_size = calculate_buffer_size ();

  if (spip::Container::verbose)
    cerr << "spip::ContainerRAM::resize size=" << size << " required_size=" << required_size << endl;

  if (required_size > size)
  {
    if (spip::Container::verbose)
      cerr << "spip::ContainerRAM::resize resizing from " << size 
           << " to " << required_size << " bytes" << endl;
    if (buffer)
    {
      // unregister the buffer prior to resizing
      if (buffer_registered)
      {
        unregister_buffer();
      }
      free (buffer);
    }
    buffer = (unsigned char *) malloc (required_size);
    if (!buffer)
      throw Error(InvalidState, "spip::ContainerRAM::resize", "malloc failed");

    size = required_size;
    if (buffer_should_register)
    {
      register_buffer();
    }
  }

  // ensure strides are correctly calculated
  if (spip::Container::verbose)
    cerr << "spip::ContainerRAM::resize calculate_strides" << endl;
  calculate_strides ();
}

void spip::ContainerRAM::zero ()
{
  bzero (buffer, size);
}

void spip::ContainerRAM::register_buffer ()
{
#if HAVE_CUDA
  unsigned int flags = 0;

  if (buffer)
  {
    // lock the data buffer block buffer as cuda memory
    cudaError_t rval = cudaHostRegister ((void *) buffer, size, flags);
    if (rval != cudaSuccess)
      throw Error (InvalidState, "spip::ContainerRAM::register_buffer", "cudaHostRegister failed");
    buffer_registered = true;
  }
  buffer_should_register = true;
#endif
}

void spip::ContainerRAM::unregister_buffer ()
{
#if HAVE_CUDA
  if (buffer)
  {
    cudaError_t rval = cudaHostUnregister ((void *) buffer);
    if (rval != cudaSuccess)
      throw Error (InvalidState, "spip::ContainerRAM::unregister_buffer", "cudaHostUnregister failed");
  }
  buffer_registered = false;
#endif
}

