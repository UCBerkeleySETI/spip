/***************************************************************************
 *
 *   Copyright (C) 2016 Andrew Jameson
 *   Licensed under the Academic Free License version 2.1
 *
 ***************************************************************************/

#include "spip/BatchedBackwardFFTCUDA.h"
#include "spip/CUFFTError.h"

#include <stdexcept>
#include <cmath>

using namespace std;

spip::BatchedBackwardFFTCUDA::BatchedBackwardFFTCUDA (cudaStream_t _stream)
{
  stream = _stream;
  plan = 0;
  nbatch = 0;
  work_area_size = 0;
  auto_allocate = false;
  work_area = 0;
}

spip::BatchedBackwardFFTCUDA::~BatchedBackwardFFTCUDA ()
{
  if (plan)
    cufftDestroy (plan);
  plan = 0;
}

void spip::BatchedBackwardFFTCUDA::prepare ()
{
  spip::BatchedBackwardFFT::prepare ();
}

void spip::BatchedBackwardFFTCUDA::prepare_plan (uint64_t ndat)
{
  if (verbose)
    cerr << "spip::BatchedBackwardFFTCUDA::prepare_plan ndat=" << ndat << endl;

  if (ndat == 0)
  {
    return;
  }

  // destroy the plan if previously allocated
  if (plan)
    cufftDestroy (plan);
  plan = 0;

  cufftResult result = cufftCreate (&plan);
  if (result != CUFFT_SUCCESS)
    throw CUFFTError (result, "spip::BatchedBackwardFFTCUDA::prepare_plan",
                      "cufftCreate(plan)");
  int rank = 1;
  int n[1] = { nfft };
  int howmany = nchan / nfft;

  int inembed[1] = { nfft };
  int istride = npol * nsignal;      // stride between samples
  int idist = nfft * istride;        // stride between FFT blocks

  int onembed[1] = { nfft };
  int ostride = 1;                   // output stride between samples
  int odist = ndat * nfft * npol;    // output stride between FFT blocks

  if (verbose)
  {
    cerr << "spip::BatchedBackwardFFTCUDA::prepare_plan howmany=" << howmany << endl;
    cerr << "spip::BatchedBackwardFFTCUDA::prepare_plan istride=" << istride << " idist=" << idist << endl;
    cerr << "spip::BatchedBackwardFFTCUDA::prepare_plan ostride=" << ostride << " odist=" << odist << endl;
  }

  size_t work_area_size;
  result = cufftMakePlanMany (plan, rank, n,
                              inembed, istride, idist,
                              onembed, ostride, odist,
                              CUFFT_C2C, howmany, &work_area_size);
  if (result != CUFFT_SUCCESS)
    throw CUFFTError (result, "BatchedBackwardFFTCUDA::prepare_plan", "cufftMakePlanMany (plan)");

  result = cufftSetStream(plan, stream);
  if (result != CUFFT_SUCCESS)
    throw CUFFTError (result, "BatchedBackwardFFTCUDA::prepare_plan", "cufftSetStream(plan)");

  // get an estimate on the work buffer size
  work_area_size = 0;
  result = cufftEstimateMany(rank, n,
                             inembed, istride, idist,
                             onembed, ostride, odist,
                             CUFFT_C2C, howmany, &work_area_size);
  if (result != CUFFT_SUCCESS)
    throw CUFFTError (result, "BatchedBackwardFFTCUDA::prepare_plan", "cufftEstimateMany");

  if (verbose)
    cerr << "BatchedBackwardFFTCUDA::prepare_plan work_area_size=" << work_area_size << endl;
  auto_allocate = work_area_size > 0;

  if (verbose)
    cerr << "BatchedBackwardFFTCUDA::prepare_plan auto_allocate=" << auto_allocate << endl;

  result = cufftSetAutoAllocation(plan, auto_allocate);
  if (result != CUFFT_SUCCESS)
    throw CUFFTError (result, "BatchedBackwardFFTCUDA::prepare_plan", "cufftSetAutoAllocation");

  if (work_area_size > 0)
  {
    cudaError_t error;
    if (work_area)
    {
      error = cudaFree (work_area);
      if (error != cudaSuccess)
        throw runtime_error("BatchedBackwardFFTCUDA::prepare_plan cudaFree(work_area) failed");
    }

    error = cudaMalloc (&work_area, work_area_size);
    if (error != cudaSuccess)
      throw runtime_error("BatchedBackwardFFTCUDA::prepare_plan cudaMalloc (work_area) failed");
  }
  else
    work_area = 0;

  nbatch = nchan / nfft;
}

//! perform BatchedBackward FFT using CUFFT
void spip::BatchedBackwardFFTCUDA::transform ()
{
  // check that the batching length is correct
  if (nbatch != nchan / nfft)
    prepare_plan (ndat); 

  cufftComplex * in  = (cufftComplex *) input->get_buffer();
  cufftComplex * out = (cufftComplex *) output->get_buffer();

  // transform ordering from TFPS -> SFPT
  const uint64_t isamp_stride = nchan * npol * nsignal;
  const uint64_t ndat_out = ndat * nfft;
 
  if (verbose)
    cerr << "spip::BatchedBackwardFFTCUDA::transform ndat=" << ndat << " ndat_out=" << ndat_out << endl;
 
  for (uint64_t idat=0; idat<ndat; idat++)
  { 
    unsigned ipolsig = 0;
    unsigned opolsig = 0; 
    for (unsigned ipol=0; ipol<npol; ipol++)
    { 
      for (unsigned isig=0; isig<nsignal; isig++)
      { 
        cufftResult result = cufftExecC2C(plan, in, out, CUFFT_FORWARD);
        if (result != CUFFT_SUCCESS)
          throw CUFFTError (result, "spip::BatchedBackwardFFTCUDA::transform", "cufftExecC2C(plan)");
        ipolsig += 1;
        opolsig += ndat_out;
      }
    }
    
    in += isamp_stride;
    out += nfft;
  }
}

