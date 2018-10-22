/***************************************************************************
 *
 *   Copyright (C) 2018 Andrew Jameson
 *   Licensed under the Academic Free License version 2.1
 *
 ***************************************************************************/

#include "spip/ContainerFile.h"
#include <unistd.h>

using namespace std;

spip::ContainerFile::ContainerFile ()
{
  fd = -1;
}

spip::ContainerFile::~ContainerFile ()
{
}

void spip::ContainerFile::close_file()
{
  if (fd >= 0)
    ::close (fd);
  fd = -1;
}

