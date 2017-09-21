/***************************************************************************
 *
 *   Copyright (C) 2016 Andrew Jameson
 *   Licensed under the Academic Free License version 2.1
 *
 ***************************************************************************/

#include "spip/Error.h"
#include "spip/ContainerCUDA.h"

#include <iostream>

using namespace std;

spip::ContainerCUDADevice::ContainerCUDADevice ()
{
  buffer = 0;
}

spip::ContainerCUDADevice::~ContainerCUDADevice ()
{
  if (buffer)
    cudaFree (buffer);
}

void spip::ContainerCUDADevice::resize ()
{
  uint64_t required_size = calculate_buffer_size ();
  if (required_size > size)
  {
    if (spip::Container::verbose)
      cerr << "spip::ContainerCUDADevice::resize allocating memory " << required_size << " bytes" << endl;
    if (buffer)
      cudaFree (buffer);
    cudaError_t err = cudaMalloc (&buffer, required_size);
    if (err != cudaSuccess)
      throw Error(InvalidState, "spip::ContainerCUDADevice::resize", cudaGetErrorString (err));

    // TODO check error
    size = required_size;
  }
}

spip::ContainerCUDAPinned::ContainerCUDAPinned ()
{
}

spip::ContainerCUDAPinned::~ContainerCUDAPinned ()
{
  // deallocate any buffer
  if (buffer)
    cudaFreeHost (buffer);
}

void spip::ContainerCUDAPinned::resize ()
{
  uint64_t required_size = calculate_buffer_size ();
  if (required_size > size)
  {
    if (spip::Container::verbose)
      cerr << "spip::ContainerCUDAPinned::resize allocating memory " << required_size << " bytes" << endl;
    if (buffer)
      cudaFreeHost(buffer);
    cudaError_t err = cudaMallocHost((void **) &buffer, required_size);
    if (err != cudaSuccess)
      throw Error(InvalidState, "spip::ContainerCUDAPinned::resize", cudaGetErrorString (err));
    size = required_size;
  }
}
