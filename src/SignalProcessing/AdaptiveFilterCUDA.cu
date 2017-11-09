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

void spip::AdaptiveFilterCUDA::set_input_rfi (Container * _input_rfi)
{
  input_rfi = dynamic_cast<spip::ContainerCUDADevice *>(_input_rfi);
  if (!input_rfi)
    throw Error (InvalidState, "spip::AdaptiveFilterCUDA::set_input_rfi",
                 "RFI input was not castable to spip::ContainerCUDADevice *");
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

void spip::AdaptiveFilterCUDA::transform_SFPT()
{
  if (verbose)
    cerr << "spip::AdaptiveFilterCUDA::transform_SFPT ()" << endl;
}

