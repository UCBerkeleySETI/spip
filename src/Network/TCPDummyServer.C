/***************************************************************************
 *
 *   Copyright (C) 2015 Andrew Jameson
 *   Licensed under the Academic Free License version 2.1
 *
 ***************************************************************************/

#include "spip/TCPDummyServer.h"

#include <unistd.h>

#include <iostream>

using namespace std;

spip::TCPDummyServer::TCPDummyServer ()
{
}

spip::TCPDummyServer::~TCPDummyServer ()
{
}

bool spip::TCPDummyServer::serve (int port)
{
  // open a listen sock on all interfaces for the control port
  if (verbose)
    cerr << "spip::TCPDummyServer::serve open socket on port=" << port << endl;
  open ("any", port, 1);

  int fd = -1;

  // wait for a connection
  while (cmd != Quit && fd < 0)
  {
    if (verbose > 1)
      cerr << "spip::TCPDummyServer::serve accept_client(1)" << endl;
    // try to accept with a 1 second timeout
    fd = accept_client (1);
    if (verbose > 1)
      cerr << "spip::TCPDummyServer::serve accept_client(1) returned fd=" << fd << endl;
    if (fd >= 0 )
    {
      if (verbose)
        cerr << "spip::TCPDummyServer::serve accepted connection, reading from socket" << endl;
      // receive a control string
      string received = read_client (65536);
      if (verbose)
        cerr << "spip::TCPDummyServer::serve read from socket, closing socket" << endl;
      close_client();
      fd = -1;
    }
    else
      if (verbose > 1)
        cerr << "spip::TCPDummyServer::serve no client connected" << endl;
  }
  return true;
}

void spip::TCPDummyServer::set_control_cmd (ControlCmd _cmd)
{
  cmd = _cmd;
  if (cmd == spip::Quit)
  {
    set_ending();
  }
};


