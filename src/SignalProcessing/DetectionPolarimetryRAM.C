/***************************************************************************
 *
 *   Copyright (C) 2016 Andrew Jameson
 *   Licensed under the Academic Free License version 2.1
 *
 ***************************************************************************/

#include "spip/DetectionPolarimetryRAM.h"

#include <iostream>
#include <cstring>
#include <stdexcept>
#include <cmath>

using namespace std;

spip::DetectionPolarimetryRAM::DetectionPolarimetryRAM () : spip::DetectionPolarimetry("DetectionPolarimetryRAM")
{
}

spip::DetectionPolarimetryRAM::~DetectionPolarimetryRAM ()
{
}

void inline spip::DetectionPolarimetryRAM::cross_detect (float p_r, float p_i, float q_r, float q_i,
                                                         float * pp, float * qq, float * pq_r, float * pq_i)
{
  *pp = (p_r * p_r) + (p_i * p_i);
  *qq = (q_r * q_r) + (q_i * q_i);
  *pq_r = (p_r * q_r) + (p_i * q_i);
  *pq_i = (p_r * q_i) - (p_i * q_r);
}

void inline spip::DetectionPolarimetryRAM::stokes_detect (float p_r, float p_i, float q_r, float q_i,
                                                         float * s0, float * s1, float * s2, float * s3)
{
  const float pp = (p_r * p_r) + (p_i * p_i);
  const float qq = (q_r * q_r) + (q_i * q_i);
  *s0 = pp + qq;
  *s1  = pp - qq;
  *s2 = 2 * ((p_r * q_r) + (p_i * q_i));
  *s3 = 2 * ((p_r * q_i) + (p_i * q_r));
}


void spip::DetectionPolarimetryRAM::transform_SFPT_to_SFPT ()
{
  if (verbose)
    cerr << "spip::DetectionPolarimetryRAM::transform_SFPT_to_SFPT" << endl;

  // read complex input data from one or two polarisations and form
  // detected products of each polarisation indepdently

  float * in_p  = (float *) input->get_buffer();
  float * in_q  = in_p + (ndim * ndat);

  float * out_a = (float *) output->get_buffer();
  float * out_b = out_a + ndat;
  float * out_c = out_b + ndat;
  float * out_d = out_c + ndat;

  const unsigned nstokes = 4;
  uint64_t in_pol_stride = ndat - (npol - 1);
  uint64_t out_pol_stride = ndat * (nstokes - 1);

  uint64_t idx = 0;
  uint64_t odx = 0;

  if (state == spip::Signal::Coherence)
  {
    for (unsigned isig=0; isig<nsignal; isig++)
    {
      for (unsigned ichan=0; ichan<nchan; ichan++)
      {
        for (uint64_t idat=0; idat<ndat; idat++)
        {
          cross_detect (in_p[idx], in_p[idx+1], in_q[idx], in_q[idx+1],
                        out_a + odx, out_b + odx, out_c + odx, out_d + odx);
          idx += 2;
          odx++;
        }
        idx += in_pol_stride;
        odx += out_pol_stride;
      }
    }
  }

  if (state == spip::Signal::Stokes)
  {
    for (unsigned isig=0; isig<nsignal; isig++)
    { 
      for (unsigned ichan=0; ichan<nchan; ichan++)
      {
        for (uint64_t idat=0; idat<ndat; idat++)
        {
          stokes_detect (in_p[idx], in_p[idx+1], in_q[idx], in_q[idx+1],
                      out_a + odx, out_b + odx, out_c + odx, out_d + odx);
          idx += 2;
          odx++;
        }
        idx += in_pol_stride;
        odx += out_pol_stride;
      }
    }
  }
}


void spip::DetectionPolarimetryRAM::transform_TSPF_to_TSPF ()
{
  if (verbose)
    cerr << "spip::DetectionPolarimetryRAM::transform_TSPF_to_TSPF()" << endl;

  float * in_p  = (float *) input->get_buffer();
  float * in_q  = in_p + (ndim * nchan);

  float * out_a = (float *) output->get_buffer();
  float * out_b = out_a + nchan;
  float * out_c = out_b + nchan;
  float * out_d = out_c + nchan;

  const unsigned nstokes = 4;
  uint64_t in_pol_stride = nchan - (npol - 1);
  uint64_t out_pol_stride = nchan * (nstokes - 1);

  uint64_t idx = 0;
  uint64_t odx = 0;

  if (state == spip::Signal::Coherence)
  {
    for (uint64_t idat=0; idat<ndat; idat++)
    {
      for (unsigned isig=0; isig<nsignal; isig++)
      {
        for (unsigned ichan=0; ichan<nchan; ichan++)
        {
          cross_detect (in_p[idx], in_p[idx+1], in_q[idx], in_q[idx+1],
                        out_a + odx, out_b + odx, out_c + odx, out_d + odx);
          idx += 2;
          odx++;
        }
        idx += in_pol_stride;
        odx += out_pol_stride;
      }
    }
  }

  if (state == spip::Signal::Stokes)
  {
    for (uint64_t idat=0; idat<ndat; idat++)
    {
      for (unsigned isig=0; isig<nsignal; isig++)
      {
        for (unsigned ichan=0; ichan<nchan; ichan++)
        {
          stokes_detect (in_p[idx], in_p[idx+1], in_q[idx], in_q[idx+1],
                      out_a + odx, out_b + odx, out_c + odx, out_d + odx);
          idx += 2;
          odx++;
        } 
        idx += in_pol_stride;
        odx += out_pol_stride;
      }
    }
  }
}

void spip::DetectionPolarimetryRAM::transform_TSPFB_to_TSPFB ()
{
  if (verbose)
    cerr << "spip::DetectionPolarimetryRAM::transform_TSPFB_to_TSPFB()" << endl;

  float * in_p  = (float *) input->get_buffer();
  float * in_q  = in_p + (ndim * nchan);

  float * out_a = (float *) output->get_buffer();
  float * out_b = out_a + nchan;
  float * out_c = out_b + nchan;
  float * out_d = out_c + nchan;

  const unsigned nstokes = 4;
  uint64_t in_pol_stride = nchan - (npol - 1);
  uint64_t out_pol_stride = nchan * (nstokes - 1);

  uint64_t idx = 0;
  uint64_t odx = 0;

  if (state == spip::Signal::Coherence)
  {
    for (uint64_t idat=0; idat<ndat; idat++)
    {
      for (unsigned isig=0; isig<nsignal; isig++)
      {
        for (unsigned ichan=0; ichan<nchan; ichan++)
        {
          for (unsigned ibin=0; ibin<nbin; ibin++)
          {
            cross_detect (in_p[idx], in_p[idx+1], in_q[idx], in_q[idx+1],
                          out_a + odx, out_b + odx, out_c + odx, out_d + odx);
            idx += 2;
            odx++;
          }
        }
        idx += in_pol_stride;
        odx += out_pol_stride;
      }
    }
  }

  if (state == spip::Signal::Stokes)
  {
    for (uint64_t idat=0; idat<ndat; idat++)
    {
      for (unsigned isig=0; isig<nsignal; isig++)
      {
        for (unsigned ichan=0; ichan<nchan; ichan++)
        {
          for (unsigned ibin=0; ibin<nbin; ibin++)
          {
            stokes_detect (in_p[idx], in_p[idx+1], in_q[idx], in_q[idx+1],
                        out_a + odx, out_b + odx, out_c + odx, out_d + odx);
            idx += 2;
            odx++;
          }
        }
        idx += in_pol_stride;
        odx += out_pol_stride;
      }
    }
  }
}


void spip::DetectionPolarimetryRAM::transform_TFPS_to_TFPS ()
{
  if (verbose)
    cerr << "spip::DetectionPolarimetryRAM::transform_TFPS_to_TFPS" << endl;
  
  // read complex input data from one or two polarisations and form
  // detected products of each polarisation indepdently
 
  float * in_p  = (float *) input->get_buffer();
  float * in_q  = in_p + (ndim * nchan);
  
  float * out_a = (float *) output->get_buffer();
  float * out_b = out_a + nchan;
  float * out_c = out_b + nchan;
  float * out_d = out_c + nchan;
  
  const unsigned nstokes = 4;
  uint64_t in_pol_stride = nsignal - (npol - 1); 
  uint64_t out_pol_stride = nsignal * (nstokes - 1);
  
  uint64_t idx = 0;
  uint64_t odx = 0;
  
  if (state == spip::Signal::PPQQ)
  {
    for (uint64_t idat=0; idat<ndat; idat++)
    { 
      for (unsigned ichan=0; ichan<nchan; ichan++)
      { 
        for (unsigned isig=0; isig<nsignal; isig++)
        {
          cross_detect (in_p[idx], in_p[idx+1], in_q[idx], in_q[idx+1],
                        out_a + odx, out_b + odx, out_c + odx, out_d + odx);
          idx += 2;
          odx++;
        }
        idx += in_pol_stride;
        odx += out_pol_stride;
      }
    }
  }

  if (state == spip::Signal::Intensity)
  { 
    for (uint64_t idat=0; idat<ndat; idat++)
    {
      for (unsigned ichan=0; ichan<nchan; ichan++)
      {
        for (unsigned isig=0; isig<nsignal; isig++)
        {               
          stokes_detect (in_p[idx], in_p[idx+1], in_q[idx], in_q[idx+1],
                         out_a + odx, out_b + odx, out_c + odx, out_d + odx);
          idx += 2;
          odx++;
        }
        idx += in_pol_stride;
        odx += out_pol_stride;
      } 
    } 
  } 
}
