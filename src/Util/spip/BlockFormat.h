
#ifndef __BlockFormat_h
#define __BlockFormat_h

#include <cstdlib>
#include <inttypes.h>

#include <vector>
#include <string>

namespace spip {

  class BlockFormat {

    public:

      BlockFormat();

      virtual ~BlockFormat();

      void prepare (unsigned _nbin, unsigned _ntime, unsigned _nfreq, double freq, double bw, double tsamp);

      void reset();

      void write_histograms(std::string hg_filename);

      void write_bandpasses(std::string bp_filename);

      void write_freq_times(std::string ft_filename);

      void write_time_series(std::string ts_filename);

      void write_mean_stddevs(std::string ms_filename);

      virtual void unpack_hgft (char * buffer, uint64_t nbytes) = 0;

      virtual void unpack_ms (char * buffer, uint64_t nbytes) = 0;

      unsigned get_nbin() { return nbin; };

      void set_resolution (uint64_t _resolution) { resolution = _resolution; };

    protected:

      float scale;

      unsigned ndim;

      unsigned npol;

      // number of input channels in the data block
      unsigned nchan;

      unsigned nbit;

      double freq;

      double bw;

      // output bandwidth, may be changed by flipping sideband
      double out_bw;

      double tsamp;

      unsigned nbin;

      unsigned ntime;

      // number of output channels
      unsigned nfreq_hg;

      unsigned nfreq_ft;

      unsigned bits_per_sample;

      unsigned bytes_per_sample;

      uint64_t resolution;

      std::vector <float> sums;

      std::vector <float> means;

      std::vector <float> variances;

      std::vector <float> stddevs;

      std::vector <std::vector <std::vector <std::vector <unsigned> > > > hist;

      std::vector <std::vector <std::vector <float> > > freq_time;

      std::vector <std::vector <float> > bandpass;

      std::vector <std::vector <float> > ts_min;

      std::vector <std::vector <float> > ts_mean;

      std::vector <std::vector <float> > ts_rms;

      std::vector <std::vector <float> > ts_max;

      std::vector <std::vector <float> > ts_sum;

      std::vector <std::vector <float> > ts_sumsq;

    private:

      char * temp_file;

  };

}

#endif
