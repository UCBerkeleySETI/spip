//-*-C++-*-
/***************************************************************************
 *
 *   Copyright (C) 2018 by Andrew Jameson
 *   Licensed under the Academic Free License version 2.1
 *
 ***************************************************************************/

#ifndef __CUDAError_h
#define __CUDAError_h

#include <cuda_runtime.h>

void check_error (const char* method);
void check_error_stream (const char* method, cudaStream_t stream);

#endif
