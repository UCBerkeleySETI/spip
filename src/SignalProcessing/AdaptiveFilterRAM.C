/***************************************************************************
 *
 *   Copyright (C) 2016 Andrew Jameson
 *   Licensed under the Academic Free License version 2.1
 *
 ***************************************************************************/

#include "spip/AdaptiveFilterRAM.h"

#include <stdexcept>
#include <cmath>

using namespace std;

spip::AdaptiveFilterRAM::AdaptiveFilterRAM ()
{
}

spip::AdaptiveFilterRAM::~AdaptiveFilterRAM ()
{
}

// configure the pipeline prior to runtime
void spip::AdaptiveFilterRAM::configure ()
{
  spip::AdaptiveFilter::configure ();
}

//! no special action required
void spip::AdaptiveFilterRAM::prepare ()
{
  spip::AdaptiveFilter::prepare();
}

// convert to antenna minor order
void spip::AdaptiveFilterRAM::transform_TSPF()
{
  if (verbose)
    cerr << "spip::AdaptiveFilterRAM::transform_TSPF ()" << endl;

}
