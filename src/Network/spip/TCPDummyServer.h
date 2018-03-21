
#ifndef __TCPDummyServer_h
#define __TCPDummyServer_h

#include "spip/TCPSocketServer.h"

#include <netinet/in.h>

namespace spip {

  enum ControlCmd   { None, Start, Stop, Quit };

  class TCPDummyServer : public TCPSocketServer {

    public:

      TCPDummyServer ();

      ~TCPDummyServer ();

      bool serve (int port);

      void set_control_cmd (ControlCmd _cmd);

    private:

      ControlCmd cmd;
  };

}

#endif
