/***************************************************************************
 *
 *   Copyright (C) 2018 Andrew Jameson
 *   Licensed under the Academic Free License version 2.1
 *
 ***************************************************************************/

#include "spip/IntegrationCUDA.h"

#include <iostream>
#include <cstring>
#include <stdexcept>
#include <cmath>
#include <cstdio>

using namespace std;

spip::IntegrationCUDA::IntegrationCUDA (cudaStream_t _stream)
{
  stream = _stream;
}

spip::IntegrationCUDA::~IntegrationCUDA ()
{
  if (buffer)
    delete buffer;
  buffer = NULL;

  if (fscrunched)
    delete fscrunched;
  fscrunched = NULL;

}

// configure the pipeline prior to runtime
void spip::IntegrationCUDA::configure (spip::Ordering output_order)
{
  if (!buffer)
    buffer = new spip::ContainerCUDADevice();
  spip::Integration::configure (output_order);

  // ensure the buffer is zero
  buffer->zero();

  // now setup the fscrunched buffer, if required
  if (chan_dec > 1)
  {
    fscrunched = new spip::ContainerCUDADevice();
    fscrunched->clone_header(input->get_header());
    fscrunched->read_header();
    if (verbose)
      cerr << "spip::IntegrationCUDA::configure fscrunched->set_nchan(" << nchan / chan_dec << ")" << endl;
    fscrunched->set_nchan (nchan / chan_dec);
    if (verbose)
      cerr << "spip::IntegrationCUDA::configure fscrunched->set_ndat(" << input->get_ndat() << ")" << endl;
    fscrunched->set_ndat (input->get_ndat());
    fscrunched->write_header();
    fscrunched->resize();
  }
}

// TSPF ordered kernel: Fscrunch operation
__global__ void Integration_TSPF_FSCR_Kernel (float * in, float * out, unsigned fdec, unsigned group_size,
                                              uint64_t in_dat_stride, uint64_t in_sigpol_stride,
                                              uint64_t out_dat_stride, uint64_t out_sigpol_stride)
{
  extern __shared__ float shm_int_tsfp_fscr[];

  // group size == chan dec 

  const unsigned group_idx = threadIdx.x % group_size;
  const unsigned group_num = threadIdx.x / group_size;   // output channel offset 
  const unsigned num_groups = blockDim.x / group_size;

  unsigned isigpol = blockIdx.y;
  unsigned idat    = blockIdx.z;
  //unsigned ochan   = (blockIdx.x * group_size) + group_num;
  unsigned ochan   = (blockIdx.x * num_groups) + group_num;
  unsigned ichan   = ochan * fdec;

  uint64_t in_block_offset  = (idat * in_dat_stride) + (isigpol * in_sigpol_stride) + ichan;

  float sum = 0.0f;
  for (unsigned i=group_idx; i<fdec; i+=group_size)
  {
    float channel = in[in_block_offset + i];
    for (int offset = group_size/2; offset>0; offset /= 2)
    {
      channel += __shfl_down (channel, offset, group_size);
    }
    if (group_idx == 0)
    {
      sum = sum + channel;
    }
  }

  if (group_idx == 0)
  {
    shm_int_tsfp_fscr[group_num] = sum;
  }

  __syncthreads();

  if (threadIdx.x < num_groups)
  {
    uint64_t out_block_offset = (idat * out_dat_stride) + (isigpol * out_sigpol_stride) + (blockIdx.x * num_groups) + threadIdx.x;
    out[out_block_offset] = shm_int_tsfp_fscr[threadIdx.x];
  }

/*
  if (group_num == 0)
  {
    uint64_t out_block_offset = (idat * out_dat_stride) + (isigpol * out_sigpol_stride) + (blockIdx.x * group_size) + group_idx;
    out[out_block_offset] = shm_int_tsfp_fscr[group_idx];
  }
*/
}

// TSPF ordered kernel: TSCR operation
__global__ void Integration_TSPF_TSCR_Kernel (float * in, float * buf, float * out, 
                                              uint64_t ndat, unsigned nchan, uint64_t buffer_idat,
                                              unsigned tdec, uint64_t sigpol_stride, uint64_t dat_stride)
{
  unsigned ichan   = blockIdx.x * blockDim.x + threadIdx.x;

  if (ichan < nchan)
  {
    unsigned isigpol = blockIdx.y;
    uint64_t idx = (isigpol * sigpol_stride) + ichan;
    uint64_t odx = idx;
    uint64_t bdx = idx;

    // load in any previous value for this channel, pol and signal
    float sum = buf[bdx];

    for (uint64_t idat=0; idat<ndat; idat++)
    {
      sum += in[idx];
      buffer_idat++;

      // if this the output sub-integration is complete
      if (buffer_idat == tdec)
      {
        // write the integrated value to the output:
        out[odx] = sum;
        odx += dat_stride;
        
        // reset the internal sum
        sum = 0;
        buffer_idat = 0;
      }

      // increment to the next time sample for this channel, pol and signal
      idx += dat_stride;
    }

    // save the partial inttegration
    buf[bdx] = sum;
  }
}

void spip::IntegrationCUDA::transform_TSPF_to_TSPF ()
{
  if (verbose)
    cerr << "spip::IntegrationCUDA::transform_TSPF_to_TSPF()" << endl;

  float * in  = (float *) input->get_buffer();
  float * out = (float *) output->get_buffer();
  float * buf = (float *) buffer->get_buffer();

  if (verbose)
  {
    cerr << "spip::IntegrationCUDA::transform_TSPF_to_TSPF nchan=" << nchan << " nsignal=" << nsignal << " npol=" << npol << " ndat=" << ndat << endl;
    cerr << "spip::IntegrationCUDA::transform_TSPF_to_TSPF chan_dec=" << chan_dec << " dat_dec=" << dat_dec << endl;
    cerr << "spip::IntegrationCUDA::transform_TSPF_to_TSPF buffer->get_size()=" << buffer->get_size() << endl;
  }

  unsigned nchan_work = nchan;
  unsigned nsigpol = nsignal * npol;

  // if fscrunching is required
  if (chan_dec > 1)
  {
    float * fscr;
    if (dat_dec > 1)
      fscr = (float *) fscrunched->get_buffer();
    else
      fscr = (float *) output->get_buffer();

    unsigned nchan_out = nchan / chan_dec;
    unsigned group_size = (chan_dec < 32) ? chan_dec : 32;
    unsigned nthread = 1024;
    unsigned ngroups = nthread / group_size;
    dim3 blocks (nchan_out / ngroups, nsigpol, ndat);
    if (nchan_out % ngroups!= 0)
      blocks.x++;

    size_t shm_bytes = ngroups * sizeof(float);

    uint64_t in_sigpol_stride = nchan;
    uint64_t in_dat_stride = in_sigpol_stride * nsigpol;
    uint64_t out_sigpol_stride = nchan_out;
    uint64_t out_dat_stride = out_sigpol_stride * nsigpol;

    if (verbose) 
    {
      cerr << "spip::IntegrationCUDA::transform_TSPF_to_TSPF Integration_TSPF_FSCR_Kernel" << endl;
      cerr << "  group_size=" << group_size << " ngroups=" << ngroups << " shm_bytes=" << shm_bytes << endl;
      cerr << "  blocks=" << blocks.x << "," << blocks.y << "," << blocks.z << " nchan=" << nchan << " nthread=" << nthread << " npol=" << npol << endl;
    }

    // first perform Fscrunching
    Integration_TSPF_FSCR_Kernel<<<blocks, nthread, shm_bytes, stream>>> (in, fscr, chan_dec, group_size, in_dat_stride, in_sigpol_stride, out_dat_stride, out_sigpol_stride);

    nchan_work = nchan_out;
    in = fscr;
  }

  if (dat_dec > 1)
  {
    unsigned nthread = 1024;
    dim3 blocks (nchan_work / nthread, nsigpol, 1);
    if (nchan_work % nthread)
      blocks.x++;

    uint64_t sigpol_stride = nchan_work;
    uint64_t dat_stride = nsigpol * sigpol_stride;

    if (verbose) 
    {
      cerr << "spip::IntegrationCUDA::transform_TSPF_to_TSPF Integration_TSPF_TSCR_Kernel buffer_idat=" << buffer_idat << " ndat=" << ndat << endl;
      cerr << "spip::IntegrationCUDA::transform_TSPF_to_TSPF Integration_TSPF_TSCR_Kernel nchan_work=" << nchan_work << endl;
    }

    // then perform Tscrunching
    Integration_TSPF_TSCR_Kernel<<<blocks, nthread, 0, stream>>>(in, buf, out, ndat, nchan, buffer_idat, dat_dec, sigpol_stride, dat_stride);

    buffer_idat += ndat;
    buffer_idat = buffer_idat % dat_dec;
  }
}


void spip::IntegrationCUDA::transform_TFPS_to_TFPS ()
{
  if (verbose)
    cerr << "spip::IntegrationCUDA::transform_TFPS_to_TFPS" << endl;
  throw invalid_argument ("spip::IntegrationCUDA::transform_TFPS_to_TFPS not implemented (yet)");
}
