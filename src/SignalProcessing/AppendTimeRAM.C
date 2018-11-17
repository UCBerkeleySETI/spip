/***************************************************************************
 *
 *   Copyright (C) 2018 Andrew Jameson
 *   Licensed under the Academic Free License version 2.1
 *
 ***************************************************************************/

#include "spip/AppendTimeRAM.h"
#include "spip/Error.h"

#include <inttypes.h>
#include <iostream>
#include <cstring>
#include <stdexcept>
#include <cmath>

using namespace std;

spip::AppendTimeRAM::AppendTimeRAM () : spip::AppendTime("AppendTimeRAM")
{
}

spip::AppendTimeRAM::~AppendTimeRAM ()
{
}

void spip::AppendTimeRAM::combine_SFPT_to_SFPT ()
{
  if (verbose)
    cerr << "spip::AppendTimeRAM::combine_SFPT_to_SFPT" << endl;
  if (nbit == 8)
  {
    int8_t dummy;
    combine_sfpt_to_sfpt (dummy);
  }
  else if (nbit == 16)
  {
    int16_t dummy;
    combine_sfpt_to_sfpt (dummy);
  }
  else if (nbit == 32)
  {
    int32_t dummy;
    combine_sfpt_to_sfpt (dummy);
  }
  else
    throw Error (InvalidState, "spip::AppendTimeRAM::combine_SFPT_to_SFPT", "input nbit [%u] must be 8, 16 or 32", nbit);
}

void spip::AppendTimeRAM::combine_TSPF_to_TSPF ()
{
  if (verbose)
    cerr << "spip::AppendTimeRAM::combine_TFPS_to_TFPS" << endl;
  if (nbit == 8)
  {
    int8_t dummy;
    combine_tspf_to_tspf (dummy);
  }
  else if (nbit == 16)
  {
    int16_t dummy;
    combine_tspf_to_tspf (dummy);
  }
  else if (nbit == 32)
  {
    int32_t dummy;
    combine_tspf_to_tspf (dummy);
  }
  else
    throw Error (InvalidState, "spip::AppendTimeRAM::combine_TFPS_to_TFPS", "input nbit [%u] must be 8, 16 or 32", nbit);
}

void spip::AppendTimeRAM::combine_TSPFB_to_TSPFB ()
{
  throw Error (InvalidState, "spip::AppendTimeRAM::combine_TSPFB_to_TSPFB", "not yet implemented");
}

void spip::AppendTimeRAM::combine_TFPS_to_TFPS ()
{
  if (verbose)
    cerr << "spip::AppendTimeRAM::combine_TFPS_to_TFPS" << endl;
  if (nbit == 8)
  {
    int8_t dummy;
    combine_tfps_to_tfps (dummy);
  }
  else if (nbit == 16)
  {
    int16_t dummy;
    combine_tfps_to_tfps (dummy);
  }
  else if (nbit == 32)
  {
    int32_t dummy;
    combine_tfps_to_tfps (dummy);
  }
  else
    throw Error (InvalidState, "spip::AppendTimeRAM::combine_TFPS_to_TFPS", "input nbit [%u] must be 8, 16 or 32", nbit);
}
