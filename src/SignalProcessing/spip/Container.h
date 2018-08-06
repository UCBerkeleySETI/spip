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
#include <iostream>

#include "spip/AsciiHeader.h"
#include "spip/Time.h"
#include "spip/Types.h"

namespace spip {

  //! All Data Containers have a sample ordering 
  typedef enum { SFPT, TFPS, TSPF, TSPFB, Custom } Ordering;

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

      void set_nbin (unsigned n) { nbin = n; }
      unsigned get_nbin() { return nbin; }
      unsigned get_nbin() const { return nbin; }

      void set_nbit (unsigned n) { nbit = n; }
      unsigned get_nbit() { return nbit; }
      unsigned get_nbit() const { return nbit; }

      void set_ndat (uint64_t n) { ndat = n; }
      uint64_t get_ndat () { return ndat; }
      uint64_t get_ndat () const { return ndat; }

      void set_tsamp (double n) { tsamp = n; }
      double get_tsamp () { return tsamp; }
      double get_tsamp () const { return tsamp; }

      void set_sideband (Signal::Sideband sb) { sideband = sb; }
      Signal::Sideband get_sideband () { return sideband; }
      Signal::Sideband get_sideband () const { return sideband; }

      void set_dual_sideband (int dsb) { dual_sideband = dsb; }
      int get_dual_sideband () { return dual_sideband; }
      int get_dual_sideband () const { return dual_sideband; }

      void set_centre_freq (double n) { centre_freq = n; }
      double get_centre_freq () { return centre_freq; }
      double get_centre_freq () const { return centre_freq; }

      void set_bandwidth (double n) { bandwidth = n; }
      double get_bandwidth () { return bandwidth; }
      double get_bandwidth () const { return bandwidth; }

      void set_file_size (int64_t n) { file_size = n; compute_file_size = false; } 
      int64_t get_file_size () { return file_size; } 
      int64_t get_file_size () const { return file_size; } 

      unsigned calculate_nbits_per_sample () { return unsigned (nsignal * nchan * nbit * npol * nbin * ndim); };
      double calculate_bytes_per_second ();

      size_t get_size () { return size; };
      size_t get_size () const { return size; };

      Time * get_utc_start () { return utc_start; };
      Time * get_utc_start () const { return utc_start; };

      size_t calculate_buffer_size () { return size_t (ndat * nchan * nsignal * ndim * npol * nbin * nbit) / 8; }
      size_t calculate_buffer_size () const { return size_t (ndat * nchan * nsignal * ndim * npol * nbin * nbit) / 8; }

      void calculate_strides ();

      inline uint64_t get_pol_stride () { return pol_stride; };
      inline uint64_t get_pol_stride () const { return pol_stride; };

      inline uint64_t get_chan_stride () { return chan_stride; };
      inline uint64_t get_chan_stride () const { return chan_stride; };

      inline uint64_t get_sig_stride () { return sig_stride; };
      inline uint64_t get_sig_stride () const { return sig_stride; };

      inline uint64_t get_bin_stride () { return bin_stride; };
      inline uint64_t get_bin_stride () const { return bin_stride; };

      inline uint64_t get_dat_stride () { return dat_stride; };
      inline uint64_t get_dat_stride () const { return dat_stride; };

      void recalculate ();

      void set_order (Ordering o) { order = o; };
      Ordering get_order() { return order; };
      Ordering get_order() const { return order; };

      void set_endianness (Endian o) { endianness = o; };
      Endian get_endianness() { return endianness; };
      Endian get_endianness() const { return endianness; };

      void set_encoding (Encoding e) { encoding = e; };
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

      void describe () const {
        std::cerr << "spip::Container::describe nbit=" << nbit << std::endl;
        std::cerr << "spip::Container::describe ndim=" << ndim << std::endl;
        std::cerr << "spip::Container::describe nsignal=" << nsignal << std::endl;
        std::cerr << "spip::Container::describe npol=" << npol << std::endl;
        std::cerr << "spip::Container::describe nchan=" << nchan << std::endl;
        std::cerr << "spip::Container::describe nbin=" << nbin << std::endl;
        std::cerr << "spip::Container::describe tsamp=" << tsamp << std::endl;
        std::cerr << "spip::Container::describe bw=" << bandwidth << std::endl;
      }


      //! return a descriptive string regarding the ordering
      static std::string get_order_string (Ordering o);

      static Ordering get_order_type (std::string o);

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

      //! Number of independent signals (e.g. antenna, beams)
      unsigned nsignal;

      //! Number of polarisations
      unsigned npol;

      //! Number of dimensions to each datum
      unsigned ndim;

      //! Number of bins of phase per time sample
      unsigned nbin;

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
      double bytes_per_second;

      //! Minimum data size
      uint64_t resolution;

      //! Preferred file size
      int64_t file_size;

      //! Number of seconds of data corresponding to each "file"
      double seconds_per_file;

      //! Number of datum between bins
      uint64_t bin_stride;
      
      //! Number of datum between dats
      uint64_t dat_stride;

      //! Number of datum between pols
      uint64_t pol_stride;

      //! Number of datum between chans
      uint64_t chan_stride;

      //! Number of datum between sigs
      uint64_t sig_stride;

      //! Positive or Negative frequnecy ordering
      Signal::Sideband sideband;

      //! Dual [1] or Single [0] Sideband
      int dual_sideband;

    private:

      bool compute_file_size;

  };
}

#endif
