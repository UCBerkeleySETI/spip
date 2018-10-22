/***************************************************************************
 *
 *   Copyright (C) 2018 Andrew Jameson
 *   Licensed under the Academic Free License version 2.1
 *
 ***************************************************************************/

#ifndef __FAppendPipeline_h
#define __FAppendPipeline_h

#include "spip/AsciiHeader.h"
#include "spip/Time.h"
#include "spip/DataBlockRead.h"
#include "spip/DataBlockWrite.h"
#include "spip/AppendFrequencyRAM.h"
#include "spip/ContainerRingRead.h"
#include "spip/ContainerRingWrite.h"

#include "config.h"

#include <vector>
#include <string>

namespace spip {

  class FAppendPipeline {

    public:

      FAppendPipeline (std::vector<std::string>, std::string);

      ~FAppendPipeline ();

      void set_output_state (Signal::State);

      void configure ();

      void open ();

      void open (const char * header_str);

      void close ();

      bool process ();

      void set_verbose () { verbose = true; };

    private:

      AsciiHeader header;

      std::vector<DataBlockRead *> in_dbs;

      DataBlockWrite * out_db;

      AppendFrequencyRAM * appender;

      std::vector<ContainerRingRead *> inputs;

      ContainerRingWrite * output;

      Signal::State output_state;

      bool verbose;

      unsigned n_inputs;

  };

}

#endif
