/***************************************************************************
 *
 *   Copyright (C) 2018 Andrew Jameson
 *   Licensed under the Academic Free License version 2.1
 *
 ***************************************************************************/

#include "spip/AppendFrequencyRAM.h"
#include "spip/Error.h"

#include <iostream>
#include <cstring>
#include <stdexcept>
#include <cmath>

using namespace std;

spip::AppendFrequencyRAM::AppendFrequencyRAM () : spip::AppendFrequency("AppendFrequencyRAM")
{
}

spip::AppendFrequencyRAM::~AppendFrequencyRAM ()
{
}

void spip::AppendFrequencyRAM::combine_SFPT_to_SFPT ()
{
  if (verbose)
    cerr << "spip::AppendFrequencyRAM::combine_SFPT_to_SFPT" << endl;
  if (nbit == 8)
    combine_sfpt_to_sfpt (int8_t dummy);
  else if (nbit == 16)
    combine_sfpt_to_sfpt (int16_t dummy);
  else if (nbit == 32)
    combine_sfpt_to_sfpt (int32_t dummy);
  else
    throw Error (InvalidState, "spip::AppendFrequencyRAM::combine_SFPT_to_SFPT", "input nbit [%u] must be 8, 16 or 32", nbit);
}

void spip::AppendFrequencyRAM::combine_TSPF_to_TSPF ()
{
  throw Error (InvalidSTFPS_to_TFPState, "spip::AppendFrequencyRAM::combine_TSPF_to_TSPF", "not yet implemented");
}

void spip::AppendFrequencyRAM::combine_TSPFB_to_TSPFB ()
{
  throw Error (InvalidSTFPS_to_TFPState, "spip::AppendFrequencyRAM::combine_TSPFB_to_TSPFB", "not yet implemented");
}

void spip::AppendFrequencyRAM::combine_TFPS_to_TFPS ()
{
  if (verbose)
    cerr << "spip::AppendFrequencyRAM::combine_TFPS_to_TFPS" << endl;
  if (nbit == 8)
    combine_tfps_to_tfps (int8_t dummy);
  else if (nbit == 16)
    combine_tfps_to_tfps (int16_t dummy);
  else if (nbit == 32)
    combine_tfps_to_tfps (int32_t dummy);
  else
    throw Error (InvalidState, "spip::AppendFrequencyRAM::combine_TFPS_to_TFPS", "input nbit [%u] must be 8, 16 or 32", nbit);
}
