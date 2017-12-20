/***************************************************************************
 *
 *   Copyright (C) 2017 Andrew Jameson
 *   Licensed under the Academic Free License version 2.1
 *
 ***************************************************************************/

#include "spip/CudaClient.h"

using namespace std;

spip::CudaClient::CudaClient (int _device_id)
{
  device_id = _device_id;

  if (device_id >= 0)
  {
    cudaError_t err;

    err = cudaGetDeviceCount(&device_count);
    if (err != cudaSuccess)
      throw Error (InvalidState, "spip::CudaClient::CudaClient",
                 cudaGetErrorString (err));

    err = cudaSetDevice (device_id);
    if (err != cudaSuccess)
      throw Error (InvalidState, "spip::CudaClient::CudaClient",
        cudaGetErrorString (err));

    err = cudaStreamCreate (&stream);
    if (err != cudaSuccess)
      throw Error (InvalidState, "spip::CudaClient::CudaClient",
        cudaGetErrorString (err));

    err = cudaGetDeviceProperties(&device_prop, device_id);
    if (err != cudaSuccess)
      throw Error (InvalidState, "spip::CudaClient::CudaClient",
        cudaGetErrorString (err));
  }
}

spip::CudaClient::~CudaClient ()
{
  if (device_id >= 0)
  {
    cudaStreamDestroy (stream);
  }
}
