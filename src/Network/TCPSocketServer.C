/***************************************************************************
 *
 *   Copyright (C) 2015 Andrew Jameson
 *   Licensed under the Academic Free License version 2.1
 *
 ***************************************************************************/

#include "spip/TCPSocketServer.h"

#include <arpa/inet.h>
#include <sys/types.h>          /* See NOTES */
#include <sys/socket.h>
#include <unistd.h>

#include <cstdlib>
#include <cstring>
#include <cerrno>
#include <iostream>

#include <stdexcept>
#include <sstream>

using namespace std;

spip::TCPSocketServer::TCPSocketServer ()
{
  client_fd = 0;
  verbose = false;
  ending = false;
}

spip::TCPSocketServer::~TCPSocketServer ()
{
  close_client();
}

// open a TCP socket and listen for up to nqueued connections
void spip::TCPSocketServer::open (std::string ip_addr, int port, int nqueued)
{
  spip::TCPSocket::open (port);

  // determine if the listening socket will use a specific or any IP address
  if (ip_addr == "any")
  {
    server.sin_addr.s_addr = INADDR_ANY;
  }
  else
  {
    struct in_addr *addr;
    addr = spip::Socket::atoaddr (ip_addr.c_str());
    server.sin_addr.s_addr = addr->s_addr;
  }

  // bind socket to file descriptor
  size_t length = sizeof(struct sockaddr_in);
  if (bind(fd, (struct sockaddr *)&server, length) == -1)
  {
    stringstream ss;
    ss << "could not bind TCPSocketServer to " << ip_addr << ":" <<  port << " " << strerror(errno);
    throw runtime_error (ss.str());
  }

  // by default all sockets are blocking
  set_block();

  // listen on the file descriptor
  if (listen(fd, nqueued) < 0)
    throw runtime_error ("Could not listen on socket");

}

void spip::TCPSocketServer::close_me ()
{
  spip::TCPSocket::close_me ();
}

// enter blocking accept call
int spip::TCPSocketServer::accept_client ()
{
  // close any active client connection (if there is one)
  close_client();

  client_fd = accept (fd, (struct sockaddr *)NULL, NULL);

  if (client_fd < 0)
  {
    throw runtime_error ("could not accept connection");
  }

  return client_fd;
}

// enter non-blocking accept call
int spip::TCPSocketServer::accept_client (int timeout)
{

  // add listening socket to the FD_SET
  fd_set socks;
  FD_ZERO (&socks);
  FD_SET(fd, &socks);

  struct timeval tmout;
  tmout.tv_sec = timeout;
  tmout.tv_usec = 0;

  int readsocks = select(fd+1, &socks, (fd_set *) 0, (fd_set *) 0, &tmout);

  if (readsocks < 0)
  {
    if (ending)
      return -1;
    else
      throw runtime_error ("select failed");
  }
  else if (readsocks == 0)
  {
    return -1;
  }
  else
    return accept_client ();
}

void spip::TCPSocketServer::close_client ()
{
  if (client_fd)
    close(client_fd);
  client_fd = 0;
}

string spip::TCPSocketServer::read_client (size_t bytes_to_recv)
{
  resize (bytes_to_recv + 1);

  ssize_t bytes_read = read (client_fd, buf, bytes_to_recv);
  if (bytes_read >= 0)
    buf[bytes_read] = '\0';

  return string(buf);
}

ssize_t spip::TCPSocketServer::write_client (char * buffer, size_t bytes)
{
  size_t bytes_to_send = bytes + 2;
  resize (bytes_to_send + 1);
  strncpy (buf, buffer, bytes);
  strcat (buf, "\r\n");

  ssize_t bytes_sent = write (client_fd, buf, bytes_to_send);
  return bytes_sent;
}

void spip::TCPSocketServer::set_verbosity (int level) 
{
  if (level)
    cerr << "spip::TCPSocketServer::set_verbosity level=" << level << endl;
  verbose = level;
};
