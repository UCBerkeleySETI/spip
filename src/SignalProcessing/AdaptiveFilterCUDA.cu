/***************************************************************************
 *
 *   Copyright (C) 2016 Andrew Jameson
 *   Licensed under the Academic Free License version 2.1
 *
 ***************************************************************************/

#include "spip/AdaptiveFilterCUDA.h"

#include <stdexcept>
#include <cmath>

using namespace std;

spip::AdaptiveFilterCUDA::AdaptiveFilterCUDA ()
{
}

spip::AdaptiveFilterCUDA::~AdaptiveFilterCUDA ()
{
}

// configure the pipeline prior to runtime
void spip::AdaptiveFilterCUDA::configure ()
{
  spip::AdaptiveFilter::configure ();
}

//! no special action required
void spip::AdaptiveFilterCUDA::prepare ()
{
  spip::AdaptiveFilter::prepare();
}

// convert to antenna minor order
void spip::AdaptiveFilterCUDA::transform_TSPF()
{
  if (verbose)
    cerr << "spip::AdaptiveFilterCUDA::transform_TSPF ()" << endl;

}
