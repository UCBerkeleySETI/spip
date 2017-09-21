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
  nbatch = 0;
}

spip::ForwardFFTCUDA::~ForwardFFTCUDA ()
{
  if (plan)
    cufftDestroy (plan);
  plan = 0;
}

void spip::ForwardFFTCUDA::prepare ()
{
  spip::ForwardFFT::prepare ();
}

void spip::ForwardFFTCUDA::prepare_plan (uint64_t ndat)
{
  if (plan)
    cufftDestroy (plan);
  
  cufftResult result = cufftCreate (&plan);
  if (result != CUFFT_SUCCESS)
    throw CUFFTError (result, "spip::ForwardFFTCUDA::prepare_plan",
                      "cufftCreate(plan)");

  nbatch = ndat / nfft;

  if (verbose)
    cerr << "spip::ForwardFFTCUDA::prepare_plan ndat=" << ndat << " nfft=" << nfft 
         << " nbatch=" << nbatch << endl;

  // input ordering is SFPT -> TFPS
  // batch over multiple time samples for a given input channel, antenna and pol
  // each batch produces 1 time sample for a set of input channels 

  int rank = 1;
  int n[1] = { nfft };
  int howmany = nbatch;

  int inembed[1] = { nfft };
  int istride = 1 ;                 // stride between samples
  int idist = nfft * istride;       // stride between FFT blocks

  int onembed[1] = { nfft };
  int ostride = 1;                    // stride between samples
  int odist = nchan * npol * nsignal; // stride between FFT blocks

  size_t work_area_size;
  result = cufftMakePlanMany (plan, rank, n,
                              inembed, istride, idist,
                              onembed, ostride, odist,
                              CUFFT_C2C, howmany, &work_area_size);
  if (result != CUFFT_SUCCESS)
    throw CUFFTError (result, "ForwardFFTCUDA::prepare_plan", "cufftMakePlanMany (plan)");

  result = cufftSetStream(plan, stream);
  if (result != CUFFT_SUCCESS)
    throw CUFFTError (result, "ForwardFFTCUDA::prepare_plan", "cufftSetStream(plan)");

  // get an estimate on the work buffer size
  work_area_size = 0;
  result = cufftEstimateMany(rank, n,
                             inembed, istride, idist,
                             onembed, ostride, odist,
                             CUFFT_C2C, howmany, &work_area_size);
  if (result != CUFFT_SUCCESS)
    throw CUFFTError (result, "ForwardFFTCUDA::prepare_plan", "cufftEstimateMany");

  if (verbose)
    cerr << "ForwardFFTCUDA::prepare_plan work_area_size=" << work_area_size << endl;
  auto_allocate = work_area_size > 0;

  if (verbose)
    cerr << "ForwardFFTCUDA::prepare_plan auto_allocate=" << auto_allocate << endl;

  result = cufftSetAutoAllocation(plan, auto_allocate);
  if (result != CUFFT_SUCCESS)
    throw CUFFTError (result, "ForwardFFTCUDA::prepare_plan", "cufftSetAutoAllocation");

  if (work_area_size > 0)
  {
    cudaError_t error;
    if (work_area)
    {
      error = cudaFree (work_area);
      if (error != cudaSuccess)
        throw runtime_error("ForwardFFTCUDA::prepare_plan cudaFree(work_area) failed");
    }

    error = cudaMalloc (&work_area, work_area_size);
    if (error != cudaSuccess)
      throw runtime_error("ForwardFFTCUDA::prepare_plan cudaMalloc (work_area) failed");
  }
  else
    work_area = 0;

}

//! perform Forward FFT using CUFFT
void spip::ForwardFFTCUDA::transform ()
{
  // check that the batching length is correct
  if (nbatch != ndat / nfft)
    prepare_plan (ndat); 

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

