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

void spip::DetectionSquareLawRAM::transform_SFPT_to_TSPF ()
{
  if (verbose)
    cerr << "spip::DetectionSquareLawRAM::transform_SFPT_to_TSPF" << endl;

  // read complex input data from one or two polarisations and form
  // detected products of each polarisation indepdently

  float * in  = (float *) input->get_buffer();
  float * out = (float *) output->get_buffer();

  const uint64_t sig_stride = output->get_sig_stride();
  const uint64_t chan_stride = output->get_chan_stride();
  const uint64_t pol_stride = output->get_pol_stride();
  const uint64_t dat_stride = output->get_dat_stride();

  if (state == spip::Signal::PPQQ)
  {
    for (unsigned isig=0; isig<nsignal; isig++)
    {
      const uint64_t sig_offset = isig * sig_stride;
      for (unsigned ichan=0; ichan<nchan; ichan++)
      {
        const uint64_t chan_offset = sig_offset + ichan * chan_stride;
        for (unsigned ipol=0; ipol<npol; ipol++)
        {
          uint64_t pol_offset = chan_offset + ipol * pol_stride;
          for (uint64_t idat=0; idat<ndat; idat++)
          {
            const float re = in[0];
            const float im = in[1];

            out[pol_offset + (idat * dat_stride)] = (re * re) + (im * im);
            in += 2;
          }
        }
      }
    }
  }

  if (state == spip::Signal::Intensity)
  {
    for (unsigned isig=0; isig<nsignal; isig++)
    {
      const uint64_t sig_offset = isig * sig_stride;
      for (unsigned ichan=0; ichan<nchan; ichan++)
      {
        const uint64_t chan_offset = sig_offset + ichan * chan_stride;
        for (unsigned ipol=0; ipol<npol; ipol++)
        {
          uint64_t pol_offset = chan_offset + ipol * pol_stride;
          for (uint64_t idat=0; idat<ndat; idat++)
          {
            const float re = in[0];
            const float im = in[1];

            if (ipol == 0)
              out[pol_offset + (idat * dat_stride)] = (re * re) + (im * im);
            else
              out[pol_offset + (idat * dat_stride)] += (re * re) + (im * im);

            in += 2;
          }
        }
      }
    }
  }
}


void spip::DetectionSquareLawRAM::transform_TSPF_to_TSPF ()
{

  float * in  = (float *) input->get_buffer();
  float * out = (float *) output->get_buffer();

  if (state == spip::Signal::PPQQ)
  {
    if (verbose)
      cerr << "spip::DetectionSquareLawRAM::transform_TSPF_to_TSPF output_state=PPQQ" << endl;
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
    if (verbose)
      cerr << "spip::DetectionSquareLawRAM::transform_TSPF_to_TSPF output_state=Intensity" << endl;
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

/*
            if (ichan == 0)
              printf ("idat=%lu idx=%lu (%f, %f) odx=%lu sum=%f\n", idat, uint64_t(in-(float *) input->get_buffer()), re, im, 
                     uint64_t(out-(float *) output->get_buffer()), *out);
*/

            in += 2;
            out++;
          }
        }
      }
    }
  }
}


void spip::DetectionSquareLawRAM::transform_TSPFB_to_TSPFB ()
{
  if (verbose)
    cerr << "spip::DetectionSquareLawRAM::transform_TSPFB_to_TSPFB()" << endl;

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
            for (unsigned ibin=0; ibin<nbin; ibin++)
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
  }

  if (state == spip::Signal::Intensity)
  {
    if (verbose)
      cerr << "spip::DetectionSquareLawRAM::transform_TSPFB_to_TSPFB ndat=" << ndat 
           << " nsignal=" << nsignal << " npol=" << npol << endl;
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
            for (unsigned ibin=0; ibin<nbin; ibin++)
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
