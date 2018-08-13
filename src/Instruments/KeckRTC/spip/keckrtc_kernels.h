/***************************************************************************
 *
 *   Copyright (C) 2018 Andrew Jameson
 *   Licensed under the Academic Free License version 2.1
 *
 ***************************************************************************/

#ifndef __keckrtc_kernels_h
#define __keckrtc_kernels_h

#include <cuda_runtime.h>

void keckrtc_dummy (void * dev_buf, size_t bufsz, cudaStream_t stream);

#endif

