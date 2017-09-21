/***************************************************************************
 *
 *   Copyright (C) 2016 Andrew Jameson
 *   Licensed under the Academic Free License version 2.1
 *
 ***************************************************************************/

#ifndef __Container_h
#define __Container_h

#include <inttypes.h>
#include <cstdlib>

#include "spip/AsciiHeader.h"
#include "spip/Time.h"

namespace spip {

  //! All Data Containers have a sample ordering 
  typedef enum { SFPT, TFPS, FSTP, Custom } Ordering;

  class Container
  {
    public:

      static bool verbose;

      //! Null constructor
      Container ();

      ~Container();

      void set_nchan (unsigned n) { nchan = n; }
      unsigned get_nchan () { return nchan; }
      unsigned get_nchan () const { return nchan; }

      void set_nsignal (unsigned n) { nsignal = n; }
      unsigned get_nsignal() { return nsignal; }
      unsigned get_nsignal() const { return nsignal; }

      void set_ndim (unsigned n) { ndim = n; }
      unsigned get_ndim() { return ndim; }
      unsigned get_ndim() const { return ndim; }

      void set_npol (unsigned n) { npol = n; }
      unsigned get_npol() { return npol; }
      unsigned get_npol() const { return npol; }

      void set_nbit (unsigned n) { nbit = n; }
      unsigned get_nbit() { return nbit; }
      unsigned get_nbit() const { return nbit; }

      void set_ndat (uint64_t n) { ndat = n; }
      uint64_t get_ndat () { return ndat; }
      uint64_t get_ndat () const { return ndat; }

      void set_tsamp (double n) { tsamp = n; }
      double get_tsamp () { return tsamp; }
      double get_tsamp () const { return tsamp; }

      void set_centre_freq (double n) { centre_freq = n; }
      double get_centre_freq () { return centre_freq; }
      double get_centre_freq () const { return centre_freq; }

      void set_bandwidth (double n) { bandwidth = n; }
      double get_bandwidth () { return bandwidth; }
      double get_bandwidth () const { return bandwidth; }

      size_t calculate_buffer_size () { return size_t (ndat * nchan * nsignal * ndim * npol * nbit) / 8; }
      size_t calculate_buffer_size () const { return size_t (ndat * nchan * nsignal * ndim * npol * nbit) / 8; }

      void set_order (Ordering o) { order = o; };
      Ordering get_order() { return order; };
      Ordering get_order() const { return order; };

      //! resize the buffer to match the input dimensions
      virtual void resize () = 0;

      //! return a pointer to the data buffer
      virtual unsigned char * get_buffer() { return buffer; }
      virtual unsigned char * get_buffer() const { return buffer; }

      // copy the meta-data from the supplied header
      void clone_header (const spip::AsciiHeader &obj);

      //! read the required meta data from the Ascii Header
      void read_header();

      //! write the meta-data to the Ascii Header
      void write_header();

      //! return const header
      AsciiHeader get_header () const { return header; } 

    protected:

      //! The data buffer 
      unsigned char * buffer;

      //! The metadata that describes the buffer
      AsciiHeader header;

      //! Size of the data buffer (in bytes)
      uint64_t size;

      //! Ordering of data within the buffer
      Ordering order;

      //! Number of time samples
      uint64_t ndat;

      //! Number of frequnecy channels
      unsigned nchan;

      //! Number of indepdent signals (e.g. antenna, beams)
      unsigned nsignal;

      //! Number of polarisations
      unsigned npol;

      //! Number of dimensions to each datum
      unsigned ndim;

      //! Number of bits per value
      unsigned nbit;

      //! Centre frequnecy of the data in MHz
      double tsamp;

      //! Centre frequnecy of the data in MHz
      double centre_freq;

      //! Bandwidth of the data in MHz
      double bandwidth;

      //! Oversampling ratio of the data (numerator / denominator)
      unsigned oversampling_ratio[2];

      //! UTC start second of the data stream
      spip::Time * utc_start;

      //! UTC start fractional second of the data stream in picoseconds
      uint64_t utc_start_pico;

      //! Offset in bytes of the data stream from the start utc
      uint64_t obs_offset;
      
    private:

  };
}

#endif
