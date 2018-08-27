/***************************************************************************
 *
 *   Copyright (C) 2018 Andrew Jameson
 *   Licensed under the Academic Free License version 2.1
 *
 ***************************************************************************/

#include "spip/File.h"

#include <unistd.h>

using namespace std;

spip::File::File ()
{
  fd = -1;
}

spip::File::~File ()
{
}

void spip::File::close_file()
{
  if (fd >= 0)
    ::close (fd);
  fd = -1;
}

