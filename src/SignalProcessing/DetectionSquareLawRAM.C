/***************************************************************************
 *
 *   Copyright (C) 2016 Andrew Jameson
 *   Licensed under the Academic Free License version 2.1
 *
 ***************************************************************************/

#include "spip/DetectionSquareLawRAM.h"

#include <iostream>
#include <cstring>
#include <stdexcept>
#include <cmath>

using namespace std;

spip::DetectionSquareLawRAM::DetectionSquareLawRAM () : spip::DetectionSquareLaw("DetectionSquareLawRAM")
{
}

spip::DetectionSquareLawRAM::~DetectionSquareLawRAM ()
{
}

void spip::DetectionSquareLawRAM::transform_SFPT_to_SFPT ()
{
  if (verbose)
    cerr << "spip::DetectionSquareLawRAM::transform_SFPT_to_SFPT" << endl;

  // read complex input data from one or two polarisations and form
  // detected products of each polarisation indepdently

  float * in  = (float *) input->get_buffer();
  float * out = (float *) output->get_buffer();

  if (state == spip::Signal::PPQQ)
  {
    for (unsigned isig=0; isig<nsignal; isig++)
    {
      for (unsigned ichan=0; ichan<nchan; ichan++)
      {
        for (unsigned ipol=0; ipol<npol; ipol++)
        {
          for (uint64_t idat=0; idat<ndat; idat++)
          {
            const float re = in[0];
            const float im = in[1];

            *out = (re * re) + (im * im);

            in += 2;
            out++;
          }
        }
      }
    }
  }

  if (state == spip::Signal::Intensity)
  {
    for (unsigned isig=0; isig<nsignal; isig++)
    { 
      for (unsigned ichan=0; ichan<nchan; ichan++)
      {
        for (unsigned ipol=0; ipol<npol; ipol++)
        {
          if (ipol == 1)
            out -= ndat;
          for (uint64_t idat=0; idat<ndat; idat++)
          {
            const float re = in[0];
            const float im = in[1];

            if (ipol == 0)
              *out = (re * re) + (im * im);
            else
              *out += (re * re) + (im * im);

            in += 2;
            out++;
          }
        }
      }
    }
  }
}


void spip::DetectionSquareLawRAM::transform_TSPF_to_TSPF ()
{
  if (verbose)
    cerr << "spip::DetectionSquareLawRAM::transform_TSPF_to_TSPF()" << endl;

  float * in  = (float *) input->get_buffer();
  float * out = (float *) output->get_buffer();

  if (state == spip::Signal::PPQQ)
  {
    for (uint64_t idat=0; idat<ndat; idat++)
    {
      for (unsigned isig=0; isig<nsignal; isig++)
      {
        for (unsigned ipol=0; ipol<npol; ipol++)
        {
          for (unsigned ichan=0; ichan<nchan; ichan++)
          {
            const float re = in[0];
            const float im = in[1];

            *out = (re * re) + (im * im);

            in += 2;
            out++;
          }
        }
      }
    }
  }

  if (state == spip::Signal::Intensity)
  {
    cerr << "spip::DetectionSquareLawRAM::transform_TSPF_to_TSPF ndat=" << ndat << " nsignal=" << nsignal << " npol=" << npol << endl;
    for (uint64_t idat=0; idat<ndat; idat++)
    {
      for (unsigned isig=0; isig<nsignal; isig++)
      {
        for (unsigned ipol=0; ipol<npol; ipol++)
        {
          if (ipol == 1)
            out -= nchan;
          for (unsigned ichan=0; ichan<nchan; ichan++)
          {
            const float re = in[0];
            const float im = in[1];

            if (ipol == 0)
              *out = (re * re) + (im * im);
            else
              *out += (re * re) + (im * im);

            if (ichan == 0)
              printf ("idat=%lu idx=%lu (%f, %f) odx=%lu sum=%f\n", idat, uint64_t(in-(float *) input->get_buffer()), re, im, 
                     uint64_t(out-(float *) output->get_buffer()), *out);

            in += 2;
            out++;
          }
        }
      }
    }
  }
}


void spip::DetectionSquareLawRAM::transform_TFPS_to_TFPS ()
{
  if (verbose)
    cerr << "spip::DetectionSquareLawRAM::transform_TFPS_to_TFPS" << endl;
  
  // read complex input data from one or two polarisations and form
  // detected products of each polarisation indepdently
  
  float * in  = (float *) input->get_buffer();
  float * out = (float *) output->get_buffer();
  
  if (state == spip::Signal::PPQQ)
  {
    for (uint64_t idat=0; idat<ndat; idat++)
    { 
      for (unsigned ichan=0; ichan<nchan; ichan++)
      { 
        for (unsigned ipol=0; ipol<npol; ipol++)
        { 
          for (unsigned isig=0; isig<nsignal; isig++)
          {  
            const float re = in[0];
            const float im = in[1];
          
            *out = (re * re) + (im * im);
          
            in += 2;
            out++;
          }
        }
      }
    }
  }
  
  if (state == spip::Signal::Intensity)
  { 
    for (uint64_t idat=0; idat<ndat; idat++)
    {
      for (unsigned ichan=0; ichan<nchan; ichan++)
      {
        for (unsigned ipol=0; ipol<npol; ipol++)
        {
          if (ipol == 1)
            out -= nsignal;
          for (unsigned isig=0; isig<nsignal; isig++)
          {
            const float re = in[0];
            const float im = in[1];

            if (ipol == 0) 
              *out = (re * re) + (im * im);
            else
              *out += (re * re) + (im * im);

            in += 2;
            out++;
          }
        }
      }
    }
  }
}
