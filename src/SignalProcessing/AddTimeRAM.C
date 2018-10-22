/***************************************************************************
 *
 *   Copyright (C) 2018 Andrew Jameson
 *   Licensed under the Academic Free License version 2.1
 *
 ***************************************************************************/

#include "spip/AddTimeRAM.h"

#include <iostream>
#include <cstring>
#include <stdexcept>
#include <cmath>

using namespace std;

spip::AddTimeRAM::AddTimeRAM () : spip::AddTime("AddTimeRAM")
{
}

spip::AddTimeRAM::~AddTimeRAM ()
{
}

void spip::AddTimeRAM::configure (spip::Ordering output_order)
{
  spip::AddTime::configure (output_order);
  input_buffers.resize(inputs.size());
}

void spip::AddTimeRAM::combine_SFPT_to_SFPT ()
{
  if (verbose)
    cerr << "spip::AddTimeRAM::combine_SFPT_to_SFPT" << endl;

  for (unsigned i=0; i<inputs.size(); i++)
    input_buffers[i] = (float *) inputs[i]->get_buffer();
  float * out = (float *) output->get_buffer();

  uint64_t idx = 0;
  for (unsigned isig=0; isig<nsignal; isig++)
  { 
    for (unsigned ichan=0; ichan<nchan; ichan++)
    {
      for (unsigned ipol=0; ipol<npol; ipol++)
      {
        for (uint64_t idat=0; idat<ndat; idat++)
        {
          for (unsigned idim=0; idim<ndim; idim++)
          {
            float sum = input_buffers[0][idx];
            for (unsigned i=1; i<inputs.size(); i++)
              out[idx] += input_buffers[i][idx];
            out[idx] = sum;
            idx++;
          }
        }
      }
    }
  }
}

void spip::AddTimeRAM::combine_TSPF_to_TSPF ()
{
  if (verbose)
    cerr << "spip::AddTimeRAM::combine_TSPF_to_TSPF" << endl;

  for (unsigned i=0; i<inputs.size(); i++)
    input_buffers[i] = (float *) inputs[i]->get_buffer();
  float * out = (float *) output->get_buffer();

  uint64_t idx = 0;
  for (uint64_t idat=0; idat<ndat; idat++)
  {
    for (unsigned isig=0; isig<nsignal; isig++)
    {
      for (unsigned ipol=0; ipol<npol; ipol++)
      {
        for (unsigned ichan=0; ichan<nchan; ichan++)
        {
          for (unsigned idim=0; idim<ndim; idim++)
          { 
            float sum = input_buffers[0][idx];
            for (unsigned i=1; i<inputs.size(); i++)
              out[idx] += input_buffers[i][idx];
            out[idx] = sum;
            idx++;
          }
        }
      }
    }
  }
}

void spip::AddTimeRAM::combine_TSPFB_to_TSPFB ()
{
  if (verbose)
    cerr << "spip::AddTimeRAM::combine_TSPFB_to_TSPFB()" << endl;

  for (unsigned i=0; i<inputs.size(); i++)
    input_buffers[i] = (float *) inputs[i]->get_buffer();
  float * out = (float *) output->get_buffer();

  uint64_t idx = 0;
  for (uint64_t idat=0; idat<ndat; idat++)
  {
    for (unsigned isig=0; isig<nsignal; isig++)
    {
      for (unsigned ipol=0; ipol<npol; ipol++)
      {
        for (unsigned ichan=0; ichan<nchan; ichan++)
        {
          for (unsigned ibin=0; ibin<nbin; ibin++)
          {
            for (unsigned idim=0; idim<ndim; idim++)
            {
              float sum = input_buffers[0][idx];
              for (unsigned i=1; i<inputs.size(); i++)
                out[idx] += input_buffers[i][idx];
              out[idx] = sum;
              idx++;
            }
          }
        }
      }
    }
  }
}

void spip::AddTimeRAM::combine_TFPS_to_TFPS ()
{
  if (verbose)
    cerr << "spip::AddTimeRAM::combine_TFPS_to_TFPS" << endl;
  
  for (unsigned i=0; i<inputs.size(); i++)
    input_buffers[i] = (float *) inputs[i]->get_buffer();
  float * out = (float *) output->get_buffer();
  
  uint64_t idx = 0;
  for (uint64_t idat=0; idat<ndat; idat++)
  { 
    for (unsigned ichan=0; ichan<nchan; ichan++)
    { 
      for (unsigned ipol=0; ipol<npol; ipol++)
      { 
        for (unsigned isig=0; isig<nsignal; isig++)
        {  
          for (unsigned idim=0; idim<ndim; idim++)
          {
            float sum = input_buffers[0][idx];
            for (unsigned i=1; i<inputs.size(); i++)
              out[idx] += input_buffers[i][idx];
            out[idx] = sum;
            idx++;
          }
        }
      }
    }
  }
}
