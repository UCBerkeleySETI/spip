
#ifndef __UDPReceiveDBStats_h
#define __UDPReceiveDBStats_h

#include "config.h"

#include "spip/UDPReceiveDB.h"
#include "spip/BlockFormat.h"

#ifdef  HAVE_VMA
#include <mellanox/vma_extra.h>
#endif

namespace spip {

  class UDPReceiveDBStats : public UDPReceiveDB  {

    public:

      UDPReceiveDBStats (const char * key_string);

      ~UDPReceiveDBStats ();

      int configure (const char * config);

      void start_monitor () { set_control_cmd (Monitor); };

      bool receive (int core);

      void set_block_format (BlockFormat * fmt);

      void set_format (UDPFormat * fmt, UDPFormat * mon_fmt);

      void configure_stats_output (std::string dir, unsigned id);

      void control_thread ();

      void analyze_block();

      char * monitor_block;

    protected:

      unsigned nchan;

      unsigned ndim;

      unsigned nbit;

      unsigned npol;

      unsigned nbin;

      double freq;

      double bw;

      double tsamp;

    private:

      UDPFormat * monitoring_udp_format;

      BlockFormat * block_format;

      std::string stats_dir;

      unsigned stream_id;

  };

}

#endif
