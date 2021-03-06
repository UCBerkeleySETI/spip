/***************************************************************************
 *
 *   Copyright (C) 2016 Andrew Jameson
 *   Licensed under the Academic Free License version 2.1
 *
 ***************************************************************************/

#include "spip/ForwardFFTCUDA.h"
#include "spip/ContainerCUDA.h"
#include "spip/CUFFTError.h"

#include <stdexcept>
#include <cmath>
#include <cstdio>

using namespace std;

spip::ForwardFFTCUDA::ForwardFFTCUDA (cudaStream_t _stream)
{
  stream = _stream;
  plan = 0;
}

spip::ForwardFFTCUDA::~ForwardFFTCUDA ()
{
  if (plan)
    cufftDestroy (plan);
  plan = 0;
}

// configure the pipeline prior to runtime
void spip::ForwardFFTCUDA::configure (spip::Ordering output_order)
{
  if (nfft == 0)
    throw runtime_error ("ForwardFFTCUDA::configure nfft not set");

  if (!conditioned)
    conditioned = new spip::ContainerCUDADevice ();

  spip::ForwardFFT::configure (output_order);

  // build the FFT plan, with ordering SFPT -> TFPS
  configure_plan();
}

void spip::ForwardFFTCUDA::configure_plan ()
{
  if (plan)
    cufftDestroy (plan);
  plan = 0;
  
  cufftResult result = cufftCreate (&plan);
  if (result != CUFFT_SUCCESS)
    throw CUFFTError (result, "spip::ForwardFFTCUDA::configure_plan",
                      "cufftCreate(plan)");

  // disable auto-allocation
  result = cufftSetAutoAllocation(plan, 0);
  if (result != CUFFT_SUCCESS)
    throw CUFFTError (result, "ForwardFFTCUDA::prepare_plan", "cufftSetAutoAllocation");

  // configure the dimensions for the plan
  configure_plan_dimensions();

  size_t work_area_size;
  result = cufftMakePlanMany (plan, rank, n,
                              inembed, istride, idist,
                              onembed, ostride, odist,
                              CUFFT_C2C, howmany, &work_area_size);
  if (result != CUFFT_SUCCESS)
    throw CUFFTError (result, "ForwardFFTCUDA::configure_plan", "cufftMakePlanMany (plan)");

  result = cufftSetStream(plan, stream);
  if (result != CUFFT_SUCCESS)
    throw CUFFTError (result, "ForwardFFTCUDA::configure_plan", "cufftSetStream(plan)");

  // get an estimate on the work buffer size
  work_area_size = 0;
  result = cufftEstimateMany(rank, n,
                             inembed, istride, idist,
                             onembed, ostride, odist,
                             CUFFT_C2C, howmany, &work_area_size);
  if (result != CUFFT_SUCCESS)
    throw CUFFTError (result, "ForwardFFTCUDA::configure_plan", "cufftEstimateMany");

  if (verbose)
    cerr << "ForwardFFTCUDA::configure_plan work_area_size=" << work_area_size << endl;
  if (work_area_size > 0)
  {
    cudaError_t error;
    if (work_area)
    {
      error = cudaFree (work_area);
      if (error != cudaSuccess)
        throw runtime_error("ForwardFFTCUDA::configure_plan cudaFree(work_area) failed");
    }

    error = cudaMalloc (&work_area, work_area_size);
    if (error != cudaSuccess)
      throw runtime_error("ForwardFFTCUDA::configure_plan cudaMalloc (work_area) failed");

    result = cufftSetWorkArea(plan, work_area);
    if (result != CUFFT_SUCCESS)
      throw CUFFTError (result, "ForwardFFTCUDA::configure_plan", "cufftSetWorkArea");
  }
  else
    work_area = 0;
}

// only words for event NFFT
__global__ void forwardfftcuda_shift_kernel (float2 * in, float2 * out, uint64_t nval)
{
  uint64_t i = blockDim.x * blockIdx.x + threadIdx.x;
  if (i < nval)
  {
    float2 val = in[i];
    const int a = int(1) - (2 * (i & 1));
    val.x *= a;
    val.y *= a;
    out[i] = val;
  }
}

__global__ void forwardfftcuda_shift_normalize_kernel(float2 * in, float2 * out, uint64_t nval, float scale_fac)
{
  uint64_t i = blockDim.x * blockIdx.x + threadIdx.x;
  if (i < nval)
  { 
    float2 val = in[i];
    const int a = int(1) - (2 * (i & 1));
    const float f = scale_fac * float(a);
    val.x *= f;
    val.y *= f;
    out[i] = val;
  }
}

//! normalize the array by the scale factor
__global__ void forwardfftcuda_normalize_kernel (float2 * in, float2 * out, uint64_t nval, float scale)
{
  uint64_t i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < nval)
  {
    float2 val = in[i];
    val.x *= scale;
    val.y *= scale;
    out[i] = val;
  }
}

void spip::ForwardFFTCUDA::condition()
{
  cufftComplex * in  = (cufftComplex *) input->get_buffer();
  cufftComplex * out = (cufftComplex *) conditioned->get_buffer();

  uint64_t nval = ndat * nsignal * npol * nchan;
  unsigned nthreads = 1024;
  uint64_t nblocks = nval / nthreads;
  if (nval % nthreads)
    nblocks++;

  if ((apply_fft_shift) && (nfft % 2 != 0))
    throw runtime_error ("ForwardFFTCUDA::condition odd fft shift not implemented (yet)");
  if (apply_fft_shift && normalize)
  {
    if (verbose)
      cerr << "spip::ForwardFFTCUDA::condition forwardfftcuda_shift_normalize_kernel" << endl;
    forwardfftcuda_shift_normalize_kernel<<<nblocks, nthreads, 0, stream>>>(in, out, nval, scale_fac);
  }
  else if (apply_fft_shift && !normalize)
  {
    if (verbose)
      cerr << "spip::ForwardFFTCUDA::condition forwardfftcuda_shift_kernel" << endl;
    forwardfftcuda_shift_kernel<<<nblocks, nthreads, 0, stream>>>(in, out, nval);
  }
  else if (!apply_fft_shift && normalize)
  {
    if (verbose)
      cerr << "spip::ForwardFFTCUDA::condition forwardfftcuda_normalize_kernel" << endl;
    forwardfftcuda_normalize_kernel<<<nblocks, nthreads, 0, stream>>>(in, out, nval, scale_fac);
  }
  else
  {
    throw runtime_error ("ForwardFFTCUDA::transform unexpected call to condition");
  }
}

//! perform Forward FFT using CUFFT
void spip::ForwardFFTCUDA::transform_SFPT_to_TFPS ()
{
  if (verbose)
    cerr << "spip::ForwardFFTCUDA::transform_SFPT_to_TFPS()" << endl;
  cufftComplex * in, * out;
  if (apply_fft_shift || normalize)
    in = (cufftComplex *) conditioned->get_buffer();
  else
    in = (cufftComplex *) input->get_buffer();
  out = (cufftComplex *) output->get_buffer();
  cufftResult result; 

  uint64_t out_offset;

  for (unsigned isig=0; isig<nsignal; isig++)
  {
    for (unsigned ichan=0; ichan<nchan; ichan++)
    {
      // output channel is offset
      unsigned ochan = ichan * nfft;
      for (unsigned ipol=0; ipol<npol; ipol++)
      {
        out_offset = isig + (ipol * nsignal) + (ochan * npol * nsignal);
        result = cufftExecC2C(plan, in, out + out_offset, CUFFT_FORWARD);
        if (result != CUFFT_SUCCESS)
          throw runtime_error ("ForwardFFTCUDA::tranform cufftExecC2C failed");

        // increment to next input
        in += ndat;
      }
    }
  }
}

// convert to frequency minor order
void spip::ForwardFFTCUDA::transform_SFPT_to_TSPF ()
{
  if (verbose)
    cerr << "spip::ForwardFFTCUDA::transform_SFPT_to_TSPF()" << endl;
  cufftComplex * in, * out;
  if (apply_fft_shift || normalize)
    in = (cufftComplex *) conditioned->get_buffer();
  else
    in = (cufftComplex *) input->get_buffer();
  out = (cufftComplex *) output->get_buffer();
  cufftResult result;

  const uint64_t nchan_out = nchan * nfft;
  const uint64_t out_pol_stride = nchan_out;
  const uint64_t out_sig_stride = npol * out_pol_stride;

  // iterate over input ordering of SFPT -> TSPF
  for (unsigned isig=0; isig<nsignal; isig++)
  {
    const uint64_t out_sig_offset = isig * out_sig_stride;
    for (unsigned ichan=0; ichan<nchan; ichan++)
    {
      const uint64_t out_chan_offset = ichan * nfft;
      for (unsigned ipol=0; ipol<npol; ipol++)
      {
        const uint64_t out_pol_offset = ipol * out_pol_stride;

        // process ndat samples, in batches of nfft
        const uint64_t out_offset = out_sig_offset + out_chan_offset + out_pol_offset;

        result = cufftExecC2C(plan, in, out + out_offset, CUFFT_FORWARD);
        if (result != CUFFT_SUCCESS)
          throw runtime_error ("ForwardFFTCUDA::tranform cufftExecC2C failed");

        in += ndat;
      }
    }
  }
}

void spip::ForwardFFTCUDA::transform_SFPT_to_SFPT ()
{
  cufftComplex * in, * out;
  if (apply_fft_shift || normalize)
    in = (cufftComplex *) conditioned->get_buffer();
  else
    in = (cufftComplex *) input->get_buffer();
  out = (cufftComplex *) output->get_buffer();
  cufftResult result;

  if (verbose)
    cerr << "spip::ForwardFFTCUDA::transform_SFPT_to_SFPT in=" << (void *) in << " out=" << (void *) out << endl;

  const uint64_t nchan_out = nchan * nfft;
  const uint64_t out_pol_stride = nbatch;
  const uint64_t out_chan_stride = npol * out_pol_stride;
  const uint64_t out_sig_stride = nchan_out * out_chan_stride;

  // iterate over input ordering of SFPT -> SFPT
  for (unsigned isig=0; isig<nsignal; isig++)
  {
    const uint64_t out_sig_offset = isig * out_sig_stride;
    for (unsigned ichan=0; ichan<nchan; ichan++)
    {
      // output channel will be ichan * nfft
      const uint64_t out_chan_offset = ichan * nfft * out_chan_stride;
      for (unsigned ipol=0; ipol<npol; ipol++)
      {
        const uint64_t out_pol_offset = ipol * out_pol_stride;

        // process ndat samples, in batches of nfft
        const uint64_t out_offset = out_sig_offset + out_chan_offset + out_pol_offset;

        result = cufftExecC2C(plan, in, out + out_offset, CUFFT_FORWARD);
        if (result != CUFFT_SUCCESS)
          throw runtime_error ("ForwardFFTCUDA::tranform cufftExecC2C failed");

        in += ndat;
      }
    }
  }
}
