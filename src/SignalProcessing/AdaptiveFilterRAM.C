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

void spip::AdaptiveFilterRAM::set_input_rfi (Container * _input_rfi)
{
  input_rfi = dynamic_cast<spip::ContainerRAM *>(_input_rfi);
  if (!input_rfi)
    throw Error (InvalidState, "spip::AdaptiveFilterRAM::set_input_rfi", 
                 "RFI input was not castable to spip::ContainerRAM *");
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

void spip::AdaptiveFilterRAM::transform_TSPF()
{
  if (verbose)
    cerr << "spip::AdaptiveFilterRAM::transform_TSPF ()" << endl;
}

void spip::AdaptiveFilterRAM::transform_SFPT()
{
  if (verbose)
    cerr << "spip::AdaptiveFilterRAM::transform_SFPT ()" << endl;
}

