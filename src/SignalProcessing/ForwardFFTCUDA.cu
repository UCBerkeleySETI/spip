/***************************************************************************
 *
 *   Copyright (C) 2016 Andrew Jameson
 *   Licensed under the Academic Free License version 2.1
 *
 ***************************************************************************/

#include "spip/ForwardFFTCUDA.h"
#include "spip/CUFFTError.h"

#include <stdexcept>
#include <cmath>

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
  cufftSetAutoAllocation(plan, 0);

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
  auto_allocate = work_area_size > 0;

  if (verbose)
    cerr << "ForwardFFTCUDA::configure_plan auto_allocate=" << auto_allocate << endl;

  //result = cufftSetAutoAllocation(plan, auto_allocate);
  //if (result != CUFFT_SUCCESS)
  //  throw CUFFTError (result, "ForwardFFTCUDA::configure_plan", "cufftSetAutoAllocation");

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

void spip::ForwardFFTCUDA::prepare ()
{
  spip::ForwardFFT::prepare ();
}

//! perform Forward FFT using CUFFT
void spip::ForwardFFTCUDA::transform_SFPT_to_TFPS ()
{
  cufftComplex * in  = (cufftComplex *) input->get_buffer();
  cufftComplex * out = (cufftComplex *) output->get_buffer();
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
  cufftComplex * in  = (cufftComplex *) input->get_buffer();
  cufftComplex * out = (cufftComplex *) output->get_buffer();
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
  cufftComplex * in  = (cufftComplex *) input->get_buffer();
  cufftComplex * out = (cufftComplex *) output->get_buffer();
  cufftResult result;

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

//! normalize the array by the scale factor
__global__ void forwardfftcuda_normalize_kernel (float * data, uint64_t nval, float scale)
{
  unsigned idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < nval)
    data[idx] = data[idx] * scale;
}

void spip::ForwardFFTCUDA::normalize_output ()
{
  float * out = (float *) output->get_buffer();

  int nthread = 1024;
  uint64_t nval = ndat * nsignal * nchan * npol * ndim;
  int nblock = nval / nthread;
  if (nval % nthread != 0)
    nblock++;

  forwardfftcuda_normalize_kernel<<<nblock, nthread, 0, stream>>> (out, nval, scale_fac);

  // TODO add check on running of kernel 
}

