/***************************************************************************
 *
 *   Copyright (C) 2018 Andrew Jameson
 *   Licensed under the Academic Free License version 2.1
 *
 ***************************************************************************/

#include "spip/IntegrationBinnedCUDA.h"

#include <cuComplex.h>

#include <iostream>
#include <cstring>
#include <stdexcept>
#include <cmath>
#include <cstdio>

using namespace std;

spip::IntegrationBinnedCUDA::IntegrationBinnedCUDA (cudaStream_t _stream)
{
  stream = _stream;
}

spip::IntegrationBinnedCUDA::~IntegrationBinnedCUDA ()
{
  if (buffer)
    delete buffer;
  buffer = NULL;

 if (binplan)
    delete binplan;
  binplan = NULL;

  if (fscrunched)
    delete fscrunched;
  fscrunched = NULL;
}

// configure the pipeline prior to runtime
void spip::IntegrationBinnedCUDA::configure (spip::Ordering output_order)
{
  if (!buffer)
    buffer = new spip::ContainerCUDADevice();

  if (!binplan)
    binplan = new spip::ContainerCUDADevice();

  spip::IntegrationBinned::configure (output_order);

  // ensure the buffer is zero
  buffer->zero();

  if (chan_dec > 1)
  {
    fscrunched = new spip::ContainerCUDADevice();
    fscrunched->clone_header(input->get_header());
    fscrunched->read_header();
    fscrunched->set_nchan (nchan / chan_dec);
    fscrunched->set_ndat (input->get_ndat());
    fscrunched->write_header();
    fscrunched->resize();
  }
}

  
// TSPF ordered kernel: Fscrunch operation
__global__ void IntegrationBinned_binplan_Kernel (int * binplan, uint64_t ndat, uint64_t start_idat,
                                                  double tsamp, int64_t cal_epoch_delta, double cal_period,
                                                  double cal_phase, double cal_duty_cycle)
{
  const uint64_t idx = (blockIdx.x * blockDim.x) + threadIdx.x;
  if (idx >= ndat)
    return;

  const uint64_t idat = idx + start_idat;

  // convert the idat to an offset time from the utc_start [seconds]
  double idat_offset_utc_start_beg = (tsamp * idat) / 1000000;
  double idat_offset_utc_start_end = (tsamp * (idat+1)) / 1000000;

  // convert to offset from cal epoch [seconds]
  double idat_offset_cal_epoch_beg = idat_offset_utc_start_beg + (cal_epoch_delta);
  double idat_offset_cal_epoch_end = idat_offset_utc_start_end + (cal_epoch_delta);

  // convert to phase of the cal
  double phi_beg = (fmod (idat_offset_cal_epoch_beg, cal_period) / cal_period) - cal_phase;
  double phi_end = (fmod (idat_offset_cal_epoch_end, cal_period) / cal_period) - cal_phase;

  // bin 0 == OFF bin 1 == ON
  // if the starting phase is greater than the duty cycle
  int bin;
  if ((phi_beg > 0) && (phi_end < cal_duty_cycle))
    bin = 1;
  else if ((phi_beg > cal_duty_cycle) && (phi_end < 1))
    bin = 0;
  else if ((phi_beg <0 ) && (phi_end < 0))
    bin = 0;
  else
    bin = -1;

  binplan[idat-start_idat] = bin;
}

void spip::IntegrationBinnedCUDA::prepare_binplan ()
{
  binplan->set_ndat (ndat);
  binplan->resize();

  int * bp = (int *) binplan->get_buffer();

  unsigned nthread = 1024;
  unsigned blocks = ndat / nthread;
  if (ndat % nthread)
    blocks++;

   // then perform Tscrunching
  IntegrationBinned_binplan_Kernel<<<blocks, nthread, 0, stream>>>(bp, ndat, start_idat, tsamp, cal_epoch_delta, cal_period, cal_phase, cal_duty_cycle);

  start_idat += ndat;
}

// TSPF ordered kernel: Fscrunch operation [only works for FDEC=32]
__global__ void IntegrationBinned_TSPF_FSCR_Kernel (float * in, float * out, unsigned fdec, unsigned group_size,
                                                    uint64_t in_dat_stride, uint64_t in_sigpol_stride,
                                                    uint64_t out_dat_stride, uint64_t out_sigpol_stride)
{
  extern __shared__ float shm_int_tspf_fscr[];

  // groups are channel scrunches (e.g. adding together 8 adjacent channels)
  unsigned group_idx = threadIdx.x % group_size;
  unsigned group_num = threadIdx.x / group_size;   // output channel offset
  unsigned num_groups = blockDim.x / group_size;   // number of groups per block

  unsigned isigpol = blockIdx.y;
  unsigned idat    = blockIdx.z;
  unsigned ichan   = ((blockIdx.x * num_groups) + group_num) * fdec;
  unsigned ochan   = (blockIdx.x * num_groups) + threadIdx.x; 

  uint64_t in_block_offset  = (idat * in_dat_stride) + (isigpol * in_sigpol_stride) + ichan;

  // groups load the the channels to be added together, scrunch using warp shuffles
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
    shm_int_tspf_fscr[group_num] = sum;
  }

  __syncthreads();

  if (threadIdx.x < num_groups)
  {
    uint64_t out_block_offset = (idat * out_dat_stride) + (isigpol * out_sigpol_stride) + ochan;
    out[out_block_offset] = shm_int_tspf_fscr[threadIdx.x];
  }
}

// TSPFB ordered kernel: Fscrunch operation [only works for FDEC=32)
__global__ void IntegrationBinned_TSPFB_FSCR_Kernel (float * in, float * out, unsigned fdec, unsigned group_size, unsigned nbin,
                                                     uint64_t in_dat_stride, uint64_t in_sigpol_stride,
                                                     uint64_t out_dat_stride, uint64_t out_sigpol_stride)
{
  extern __shared__ float shm_int_tspfb_fscr[];

  // groups are channel scrunches (e.g. adding together 8 adjacent channels)
  unsigned group_idx = threadIdx.x % group_size;
  unsigned group_num = threadIdx.x / group_size;   // output channel offset

  unsigned isigpol = blockIdx.y;
  unsigned idat    = blockIdx.z;
  unsigned ochan   = (blockIdx.x * group_size) + group_num;
  unsigned ichan   = ochan * fdec;

  uint64_t in_block_offset  = (idat * in_dat_stride) + (isigpol * in_sigpol_stride) + (ichan * nbin);

  // groups of size FDEC, will add channels via warp shuffling

  // groups load the the channels to be added together, scrunch using warp shuffles
  float sum = 0.0f;
  for (unsigned ibin=0; ibin<nbin; ibin++)
  {
    for (unsigned i=group_idx; i<fdec; i+=group_size)
    {
      // load the channel and bin for this group
      float channel = in[in_block_offset + (i*nbin) + ibin];
      {
        for (int offset = group_size/2; offset>0; offset /= 2)
        {
          channel += __shfl_down (channel, offset, group_size);
        }
        // the lead thread of the group saves the sum
        if (group_idx == 0)
        {
          sum = sum + channel;
        }
      }
    }

    if (group_idx == 0)
    {
      shm_int_tspfb_fscr[(group_num * nbin) + ibin] = sum;
    }
  }

  __syncthreads();

  // we have group_num sums of 
  if (threadIdx.x < (group_num * nbin))
  {
    uint64_t out_block_offset = (idat * out_dat_stride) + (isigpol * out_sigpol_stride) + (blockIdx.x * group_size) + group_idx;
    out[out_block_offset] = shm_int_tspfb_fscr[group_idx];
  }
}


// TSPF ordered kernel: TSCR operation not efficient for small number of channels and lots of time samples
__global__ void IntegrationBinned_TSPF_TSPFB_TSCR_Kernel (float * in, float * buf, float * out, int * bp,
                                                          uint64_t ndat, unsigned nchan, unsigned nbin, 
                                                          uint64_t buffer_idat, unsigned tdec, 
                                                          uint64_t in_sigpol_stride, uint64_t in_dat_stride)
{
  const unsigned ichan = blockIdx.x * blockDim.x + threadIdx.x;

  if (ichan < nchan)
  {
    unsigned isigpol = blockIdx.y;
    uint64_t idx = (isigpol * in_sigpol_stride) + ichan;
    uint64_t odx = idx * nbin;
    uint64_t bdx = odx;

    // load in any previous value for this channel, pol and signal and bins [assume nbin==2]
    float sums[2];
    sums[0] = buf[bdx+0];
    sums[1] = buf[bdx+1];

    for (uint64_t idat=0; idat<ndat; idat++)
    {
      buffer_idat++;

      const int ibin = bp[idat];
      if (ibin >= 0)
      {
        sums[ibin] += in[idx];
      }

      // if this the output sub-integration is complete
      if (buffer_idat == tdec)
      {
        // write the integrated value to the output:
        out[odx + 0] = sums[0];
        out[odx + 1] = sums[1];

        odx += (in_dat_stride * nbin);
       
        // reset the internal sum
        sums[0] = 0;
        sums[1] = 0;
        buffer_idat = 0;
      }

      // increment to the next time sample for this channel, sigpol and bin
      idx += in_dat_stride;
    }

    // save the partial integration for this channel, sigpol and bin
    buf[bdx+0] = sums[0];
    buf[bdx+1] = sums[1];
  }
}

// compute a sum of a float across a warp
__inline__ __device__
cuFloatComplex warpReduceSumFC(cuFloatComplex val)
{
  for (int offset = warpSize/2; offset > 0; offset /= 2)
  {
    val.x += __shfl_down(val.x, offset);
    val.y += __shfl_down(val.y, offset);
  }
  return val;
}

// TSPF ordered kernel: TSCR operation for LargeT scrunch
__global__ void IntegrationBinned_TSPF_TSPFB_TSCR_LargeT_Kernel (float * in, float * buf, float * out, int * bp,
                                                                 uint64_t ndat, unsigned nchan, unsigned nbin,
                                                                 uint64_t buffer_idat, unsigned tdec,
                                                                 uint64_t in_sigpol_stride, uint64_t in_dat_stride)
{
  extern __shared__ float shm_int_tspf_tspfb_tscr[];

  // each block processes 32 channels and  time samples 
  // each warp will read 32 channels into SHM, then we transpose and each warp sums 32 time samples
  const unsigned warp_idx = threadIdx.x % 32; // [0..31] TODO make efficient
  const unsigned warp_num = threadIdx.x / 32; // [0..31]

  const unsigned isigpol = blockIdx.y;
  const unsigned ichan = (blockIdx.x * 32) + warp_idx;
  const unsigned ochan = (blockIdx.x * 32) + warp_num;  // only warp_idx == 0 will read/write

  uint64_t idx = (warp_num * in_dat_stride) + (isigpol * in_sigpol_stride) + ichan;
  uint64_t odx = ((isigpol * in_sigpol_stride) + ochan) * nbin;
  uint64_t bdx = odx;

  // use a cuFloatComplex to store the low/high bins
  cuFloatComplex sums = make_cuFloatComplex(0, 0);

  // load the previous bin values for these channels in the first warp only
  // use first threads in warp to load 32 values from buffer
  if (warp_idx == 0)
  {
    sums.x = buf[bdx+0];
    sums.y = buf[bdx+1];
  }

  for (uint64_t idat=warp_idx; idat<ndat; idat+=32)
  {
    // read 32 time samples for 32 channels into SHM writing in TF order
    // write 33 to eliminate warp lane clashing
    shm_int_tspf_tspfb_tscr[(warp_num * 33) + warp_idx] = in[idx];

    // wait for all warps to have loaded the 1024 samples
    __syncthreads();

    // determine the bin for this idat
    const int ibin = bp[idat];

    // now transpose so that each warp has 32 consecutive time samples for 1 channel
    float val = shm_int_tspf_tspfb_tscr[warp_idx * 33 + warp_num];

    // sum this time sample into the right bin
    if (ibin == 0)
      sums.x += val;
    else if (ibin == 1)
      sums.y += val;
 
    // now sum across the warp (32 time samples) for 1 channel
    sums = warpReduceSumFC(sums);

    // increment the number of idats counted
    buffer_idat += 32;

    // warp_idx == 0 has the sums for each channel, reset the others
    if (warp_idx != 0)
      sums = make_cuFloatComplex(0, 0);

    // if this the output sub-integration is complete
    if (buffer_idat >= tdec)
    {
      // write the integrated value to the output:
      if (warp_idx == 0)
      {
        out[odx + 0] = sums.x;
        out[odx + 1] = sums.y;
        odx += (in_dat_stride * nbin);
      }

      // reset the internal sum
      sums.x = 0;
      sums.y = 0;
      buffer_idat = 0;
    }

    idx += in_dat_stride;
  }

  // save the sum from each channel to buffer
  if (warp_idx == 0)
  {
    buf[bdx+0] = sums.x;
    buf[bdx+1] = sums.y;
  }
}

void spip::IntegrationBinnedCUDA::transform_TSPF_to_TSPFB()
{
  if (verbose)
    cerr << "spip::IntegrationBinnedCUDA::transform_TSPF_to_TSPF()" << endl;

  float * in  = (float *) input->get_buffer();
  float * out = (float *) output->get_buffer();
  int * bp = (int *) binplan->get_buffer();
  float * buf = (float *) buffer->get_buffer();

  unsigned nchan_work = nchan;
  unsigned nsigpol = nsignal * npol;

  // if fscrunching is required
  if (chan_dec > 1)
  {
    float * fscr = (float *) fscrunched->get_buffer();

    unsigned nchan_out = nchan / chan_dec;
    unsigned group_size = (chan_dec < 32) ? chan_dec : 32;
    unsigned nthread = (nchan < 1024) ? nchan : 1024;      // AJ change - check
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
      cerr << "spip::IntegrationBinnedCUDA::transform_TSPF_to_TSPFB IntegrationBinned_TSPF_FSCR_Kernel" << endl;
      cerr << "spip::IntegrationBinnedCUDA::transform_TSPF_to_TSPFB group_size=" << group_size << " ngroups=" << ngroups << " shm_bytes=" << shm_bytes << endl;
      cerr << "spip::IntegrationBinnedCUDA::transform_TSPF_to_TSPFB blocks=" << blocks.x << "," << blocks.y << "," << blocks.z << " nchan=" << nchan << " nthread=" << nthread << " npol=" << npol << endl;
      cerr << "spip::IntegrationBinnedCUDA::transform_TSPF_to_TSPFB in_sigpol_stride=" << in_sigpol_stride << " in_dat_stride=" << in_dat_stride << " out_sigpol_stride=" << out_sigpol_stride << " out_dat_stride=" << out_dat_stride << endl;
    }

    // first perform Fscrunching
    IntegrationBinned_TSPF_FSCR_Kernel<<<blocks, nthread, shm_bytes, stream>>> (in, fscr, chan_dec, group_size, in_dat_stride, in_sigpol_stride, out_dat_stride, out_sigpol_stride);

    nchan_work = nchan_out;
    in = fscr;
  }

  if (dat_dec > 1)
  {
    unsigned nthread = 1024;

    if (nchan_work > ndat)
    {
      dim3 blocks (nchan_work / nthread, nsigpol, 1);
      if (nchan_work % nthread)
        blocks.x++;

      uint64_t sigpol_stride = nchan_work;
      uint64_t dat_stride = nsigpol * sigpol_stride;

      if (verbose) 
      {
        cerr << "spip::IntegrationBinnedCUDA::transform_TSPF_to_TSPFB IntegrationBinned_TSPF_TSCR_Kernel blocks=" << blocks.x << "," << blocks.y << "," << blocks.z << endl;
        cerr << "spip::IntegrationBinnedCUDA::transform_TSPF_to_TSPFB IntegrationBinned_TSPF_TSCR_Kernel buffer_idat=" << buffer_idat << " ndat=" << ndat << endl;
        cerr << "spip::IntegrationBinnedCUDA::transform_TSPF_to_TSPFB IntegrationBinned_TSPF_TSCR_Kernel nchan_work=" << nchan_work << " dat_stride=" << dat_stride << endl;
      }

      // then perform Tscrunching
      IntegrationBinned_TSPF_TSPFB_TSCR_Kernel<<<blocks, nthread, 0, stream>>>(in, buf, out, bp, ndat, nchan_work, nbin, buffer_idat, dat_dec, sigpol_stride, dat_stride);
    }
    else
    {
      dim3 blocks (nchan_work / 32, nsigpol, 1);
      if (nchan_work % 32)
        blocks.x++;

      uint64_t sigpol_stride = nchan_work;
      uint64_t dat_stride = nsigpol * sigpol_stride;

      if (verbose) 
      {
        cerr << "spip::IntegrationBinnedCUDA::transform_TSPF_to_TSPFB IntegrationBinned_TSPF_TSCR_LargeT_Kernel blocks=" << blocks.x << "," << blocks.y << "," << blocks.z << endl;
        cerr << "spip::IntegrationBinnedCUDA::transform_TSPF_to_TSPFB IntegrationBinned_TSPF_TSCR_LargeT_Kernel buffer_idat=" << buffer_idat << " ndat=" << ndat << endl;
        cerr << "spip::IntegrationBinnedCUDA::transform_TSPF_to_TSPFB IntegrationBinned_TSPF_TSCR_LargeT_Kernel nchan_work=" << nchan_work << " dat_stride=" << dat_stride << endl;
      }

      size_t sbytes = 1056 * sizeof(float);

      // then perform Tscrunching
      IntegrationBinned_TSPF_TSPFB_TSCR_LargeT_Kernel<<<blocks, nthread, sbytes, stream>>>(in, buf, out, bp, ndat, nchan_work, nbin, buffer_idat, dat_dec, sigpol_stride, dat_stride);
    }

    buffer_idat += ndat;
    buffer_idat = buffer_idat % dat_dec;
  }
}
