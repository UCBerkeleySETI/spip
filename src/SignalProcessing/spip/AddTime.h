//-*-C++-*-
/***************************************************************************
 *
 *   Copyright (C) 2018 by Andrew Jameson
 *   Licensed under the Academic Free License version 2.1
 *
 ***************************************************************************/

#ifndef __AddTime_h
#define __AddTime_h

#include "spip/Append.h"

namespace spip {

  class AddTime: public Append
  {
    public:
     
      AddTime (const char *);

      ~AddTime ();

      void configure (Ordering output_order);

      void set_output_state (Signal::State _state);

      //! Required data combinations
      virtual void combine_SFPT_to_SFPT () = 0;

      virtual void combine_TFPS_to_TFPS () = 0;

      virtual void combine_TSPF_to_TSPF () = 0;

      virtual void combine_TSPFB_to_TSPFB () = 0;

    protected:

    private:

  };

}

#endif
