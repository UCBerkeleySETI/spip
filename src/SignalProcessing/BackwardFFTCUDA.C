/***************************************************************************
 *
 *   Copyright (C) 2016 Andrew Jameson
 *   Licensed under the Academic Free License version 2.1
 *
 ***************************************************************************/

#include "spip/BackwardFFTCUDA.h"
#include "spip/CUFFTError.h"

#include <stdexcept>
#include <cmath>

using namespace std;

spip::BackwardFFTCUDA::BackwardFFTCUDA (cudaStream_t _stream)
{
  stream = _stream;
  plan = 0;
  work_area_size = 0;
  auto_allocate = false;
  work_area = 0;
}

spip::BackwardFFTCUDA::~BackwardFFTCUDA ()
{
  if (plan)
    cufftDestroy(plan);
  plan = 0;
}

void spip::BackwardFFTCUDA::configure (spip::Ordering output_order)
{
  if (nfft == 0)
    throw runtime_error ("BackwardFFTCUDA::configure nfft not set");

  spip::BackwardFFT::configure (output_order);

  configure_plan ();
}

void spip::BackwardFFTCUDA::configure_plan ()
{
  if (verbose)
    cerr << "spip::BackwardFFTCUDA::configure_plan ndat=" << ndat << endl;

  // no function if ndat == 0
  if (ndat == 0)
    return;

  // if we are reconfiguring the batching, destroy the previous plan
  if (plan)
    cufftDestroy(plan);
  plan = 0;

  cufftResult result = cufftCreate (&plan);
  if (result != CUFFT_SUCCESS)
    throw CUFFTError (result, "spip::BackwardFFTCUDA::prepare_plan",
                      "cufftCreate(plan)");

  configure_plan_dimensions();

  size_t work_area_size;
  result = cufftMakePlanMany (plan, rank, n,
                              inembed, istride, idist,
                              onembed, ostride, odist,
                              CUFFT_C2C, howmany, &work_area_size);
  if (result != CUFFT_SUCCESS)
    throw CUFFTError (result, "BackwardFFTCUDA::prepare_plan", "cufftMakePlanMany (plan)");

  result = cufftSetStream(plan, stream);
  if (result != CUFFT_SUCCESS)
    throw CUFFTError (result, "BackwardFFTCUDA::prepare_plan", "cufftSetStream(plan)");

  // get an estimate on the work buffer size
  work_area_size = 0;
  result = cufftEstimateMany(rank, n,
                             inembed, istride, idist,
                             onembed, ostride, odist,
                             CUFFT_C2C, howmany, &work_area_size);
  if (result != CUFFT_SUCCESS)
    throw CUFFTError (result, "BackwardFFTCUDA::prepare_plan", "cufftEstimateMany");

  if (verbose)
    cerr << "BackwardFFTCUDA::prepare_plan work_area_size=" << work_area_size << endl;
  auto_allocate = work_area_size > 0;

  if (verbose)
    cerr << "BackwardFFTCUDA::prepare_plan auto_allocate=" << auto_allocate << endl;

  result = cufftSetAutoAllocation(plan, auto_allocate);
  if (result != CUFFT_SUCCESS)
    throw CUFFTError (result, "BackwardFFTCUDA::prepare_plan", "cufftSetAutoAllocation");

  if (work_area_size > 0)
  {
    cudaError_t error;
    if (work_area)
    {
      error = cudaFree (work_area);
      if (error != cudaSuccess)
        throw runtime_error("BackwardFFTCUDA::prepare_plan cudaFree(work_area) failed");
    }

    error = cudaMalloc (&work_area, work_area_size);
    if (error != cudaSuccess)
      throw runtime_error("BackwardFFTCUDA::prepare_plan cudaMalloc (work_area) failed");
  }
  else
    work_area = 0;

}


//! no special action required
void spip::BackwardFFTCUDA::prepare ()
{
  spip::BackwardFFT::prepare();
}

//! perform Backward FFT using CUDA
void spip::BackwardFFTCUDA::transform_TFPS_to_SFPT ()
{
  cufftComplex * in  = (cufftComplex *) input->get_buffer();
  cufftComplex * out = (cufftComplex *) output->get_buffer();
  cufftResult result;

  // this FFT will produce the specified number of output channels
  const uint64_t ndat_out = ndat * nfft;
  const uint64_t istride = nsignal * npol * nfft;
  const uint64_t ostride = npol * ndat_out;

  for (uint64_t idat=0; idat<ndat; idat++)
  {
    for (unsigned ochan=0; ochan<nchan_out; ochan++)
    {
      unsigned ipolsig = 0;
      unsigned opolsig = 0;
      for (unsigned ipol=0; ipol<npol; ipol++)
      {
        for (unsigned isig=0; isig<nsignal; isig++)
        {
          result = cufftExecC2C(plan, in + ipolsig, out, CUFFT_FORWARD);
          if (result != CUFFT_SUCCESS)
            throw CUFFTError (result, "spip::BackwardFFTCUDA::transform_TFPS_to_SFPT", "cufftExecC2C(plan)");

          ipolsig += 1;
          opolsig += ndat_out;
        }
      }
      in += istride;
      out += ostride;
    }
  }
}

void spip::BackwardFFTCUDA::transform_TSPF_to_SFPT ()
{
  cufftComplex * in  = (cufftComplex *) input->get_buffer();
  cufftComplex * out = (cufftComplex *) output->get_buffer();
  cufftResult result;

  const uint64_t in_chan_stride = nchan;
  const uint64_t ndat_out = ndat * nfft;
  const uint64_t out_pol_stride = ndat_out;
  const uint64_t out_chan_stride = npol * out_pol_stride;
  const uint64_t out_sig_stride = nchan_out * out_chan_stride;

  for (unsigned isig=0; isig<nsignal; isig++)
  {
    const uint64_t out_sig_offset = isig * out_sig_stride;
    for (unsigned ipol=0; ipol<npol; ipol++)
    {
      const uint64_t out_pol_offset = ipol * out_pol_stride;
      for (unsigned ochan=0; ochan<nchan_out; ochan++)
      {
        const uint64_t out_chan_offset = ochan * out_chan_stride;
        const uint64_t out_offset = out_chan_offset + out_sig_offset + out_pol_offset;

        result = cufftExecC2C(plan, in, out + out_offset, CUFFT_FORWARD);
        if (result != CUFFT_SUCCESS)
          throw CUFFTError (result, "spip::BackwardFFTCUDA::transform_TSPF_to_SFPT", "cufftExecC2C(plan)");

        in += in_chan_stride;
      }
    }
  }
}

void spip::BackwardFFTCUDA::transform_SFPT_to_SFPT ()
{
  cufftComplex * in  = (cufftComplex *) input->get_buffer();
  cufftComplex * out = (cufftComplex *) output->get_buffer();
  cufftResult result;

  const uint64_t ndat_out = ndat * nfft;
  const uint64_t out_pol_stride = ndat_out;
  const uint64_t in_pol_stride  = ndat;
  const uint64_t out_chan_stride = npol * out_pol_stride;
  const uint64_t in_chan_stride  = npol * in_pol_stride;
  const uint64_t out_sig_stride = nchan_out * out_chan_stride;
  const uint64_t in_sig_stride  = nchan * in_chan_stride;

  for (unsigned isig=0; isig<nsignal; isig++)
  {
    const uint64_t in_sig_offset  = isig * in_sig_stride;
    const uint64_t out_sig_offset = isig * out_sig_stride;

    for (unsigned ochan=0; ochan<nchan_out; ochan++)
    {
      const uint64_t in_chan_offset  = ochan * in_chan_stride;
      const uint64_t out_chan_offset = ochan * out_chan_stride;

      for (unsigned ipol=0; ipol<npol; ipol++)
      {
        const uint64_t in_pol_offset  = ipol * in_pol_stride;
        const uint64_t out_pol_offset = ipol * out_pol_stride;

        const uint64_t in_offset  = in_chan_offset + in_sig_offset + in_pol_offset;
        const uint64_t out_offset = out_chan_offset + out_sig_offset + out_pol_offset;

        result = cufftExecC2C(plan, in + in_offset, out + out_offset, CUFFT_FORWARD);
        if (result != CUFFT_SUCCESS)
          throw CUFFTError (result, "spip::BackwardFFTCUDA::transform_SFPT_to_SFPT", "cufftExecC2C(plan)");
      }
    }
  }
}
