/***************************************************************************
 *
 *   Copyright (C) 2018 by Willem van Straten
 *   Licensed under the Academic Free License version 2.1
 *
 ***************************************************************************/

#include "spip/CUDAError.h"
#include "spip/Error.h"

#include <cuda.h>
#include <iostream>

using namespace std;

void check_error (const char* method)
{
  cudaThreadSynchronize ();

  cudaError error = cudaGetLastError();
  if (error != cudaSuccess)
  {
    cerr << method << " cudaGetLastError="
         << cudaGetErrorString (error) << endl;

    throw Error (InvalidState, method, cudaGetErrorString (error));
  }
}

void check_error_stream (const char* method, cudaStream_t stream)
{
  if (!stream)
    throw Error (InvalidState, method, "called check_error_stream on invalid stream");
  else
  {
    cudaStreamSynchronize (stream);

    cudaError error = cudaGetLastError();
    if (error != cudaSuccess)
    {
      cerr << method << " cudaGetLastError="
           << cudaGetErrorString (error) << endl;

      throw Error (InvalidState, method, cudaGetErrorString (error));
    }
  }
}

