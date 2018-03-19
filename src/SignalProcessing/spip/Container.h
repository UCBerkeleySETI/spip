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
#include "spip/Types.h"

namespace spip {

  //! All Data Containers have a sample ordering 
  typedef enum { SFPT, TFPS, TSPF, Custom } Ordering;

  //! Data may be packed in big or little endian
  typedef enum { Little, Big } Endian;

  //! Data may be ordered as Offset Binary or Twos Complement
  typedef enum { OffsetBinary, TwosComplement} Encoding;

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

      void set_file_size (int64_t n) { file_size = n; compute_file_size = false; } 
      int64_t get_file_size () { return file_size; } 
      int64_t get_file_size () const { return file_size; } 

      unsigned calculate_nbits_per_sample () { return unsigned (nsignal * nchan * nbit * npol * ndim); };
      uint64_t calculate_bytes_per_second ();

      size_t get_size () { return size; };
      size_t calculate_buffer_size () { return size_t (ndat * nchan * nsignal * ndim * npol * nbit) / 8; }
      size_t calculate_buffer_size () const { return size_t (ndat * nchan * nsignal * ndim * npol * nbit) / 8; }

      void recalculate ();

      void set_order (Ordering o) { order = o; };
      Ordering get_order() { return order; };
      Ordering get_order() const { return order; };

      void set_endianness (Endian o) { endianness = o; };
      Endian get_endianness() { return endianness; };
      Endian get_endianness() const { return endianness; };

      void set_endcoding (Encoding e) { encoding = e; };
      Encoding get_encoding() { return encoding; };
      Encoding get_encoding() const { return encoding; };

      //! resize the buffer to match the input dimensions
      virtual void resize () = 0;

      //! return a pointer to the data buffer
      virtual unsigned char * get_buffer() { return buffer; }
      virtual unsigned char * get_buffer() const { return buffer; }

      //! zero the contents of the buffer
      virtual void zero () = 0;

      // copy the meta-data from the supplied header
      void clone_header (const spip::AsciiHeader &obj);

      //! read the required meta data from the Ascii Header
      void read_header();

      //! write the meta-data to the Ascii Header
      void write_header();

      //! return const header
      AsciiHeader get_header () const { return header; } 

      //! return a descriptive string regarding the ordering
      static std::string get_order_string (Ordering o);

    protected:

      //! The data buffer 
      unsigned char * buffer;

      //! The metadata that describes the buffer
      AsciiHeader header;

      //! Size of the data buffer (in bytes)
      uint64_t size;

      //! Ordering of data within the buffer
      Ordering order;

      //! Byte ordering of data samples
      Endian endianness;

      //! Bit Encoding of data samples
      Encoding encoding;

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

      //! Number of bits per sample, accounting for all dimensions
      unsigned bits_per_sample;

      //! Number of bytes per second
      uint64_t bytes_per_second;

      //! Minimum data size
      uint64_t resolution;

      //! Preferred file size
      int64_t file_size;

      //! Number of seconds of data corresponding to each "file"
      double seconds_per_file;
      
    private:

      bool compute_file_size;

  };
}

#endif
