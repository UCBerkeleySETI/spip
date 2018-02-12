/***************************************************************************
 *
 *   Copyright (C) 2017 Andrew Jameson
 *   Licensed under the Academic Free License version 2.1
 *
 ***************************************************************************/

#include "spip/DadaClient.h"

using namespace std;

spip::DadaClient::DadaClient ()
{
  transfer_bytes = 0;
  optimal_bytes = 0;
  verbose = false;
}

spip::DadaClient::~DadaClient ()
{

}

