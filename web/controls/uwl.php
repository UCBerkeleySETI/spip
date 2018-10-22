<?PHP

error_reporting(E_ALL);
ini_set("display_errors", 1);

include_once("../spip_webpage.lib.php");
include_once("../spip.lib.php");
include_once("../spip_socket.lib.php");

class uwl extends spip_webpage
{
  function uwl ()
  {
    spip_webpage::spip_webpage ();

    $this->title = "UWL FPGA Controls";
    $this->nav_item = "uwl";

    $this->config = spip::get_config();

    $this->fpga_host = "octomore.atnf.csiro.au";
    $this->fpga_port = "17099";
    $this->lock_file = "";
  }

  function javaScriptCallback()
  {
    return "control_request();";
  }

  function printJavaScriptHead()
  {
?>
    <script type='text/javascript'>

      function handle_control_request(c_xml_request)
      {
        if (c_xml_request.readyState == 4)
        {
          var xmlDoc = c_xml_request.responseXML;
          if (xmlDoc != null)
          {
            var xmlObj=xmlDoc.documentElement;

            //var http_server = xmlObj.getElementsByTagName("http_server")[0].childNodes[0].nodeValue;
            //var url_prefix  = xmlObj.getElementsByTagName("url_prefix")[0].childNodes[0].nodeValue;
            
            var request = xmlObj.getElementsByTagName("request")[0]
            var state = xmlObj.getElementsByTagName("state")[0]
            var message = xmlObj.getElementsByTagName("message")[0]
  
            document.getElementById("uwl_fpga_request").innerHTML = request.childNodes[0].nodeValue;
            document.getElementById("uwl_fpga_state").innerHTML = state.childNodes[0].nodeValue;
            document.getElementById("uwl_fpga_message").innerHTML = message.childNodes[0].nodeValue;
          }
        }
      }

      function control_request() 
      {
        var url = "?update=true";

        if (window.XMLHttpRequest)
          c_xml_request = new XMLHttpRequest();
        else
          c_xml_request = new ActiveXObject("Microsoft.XMLHTTP");

        c_xml_request.onreadystatechange = function()
        {
          handle_control_request(c_xml_request)
        };
        c_xml_request.open("GET", url, true);
        c_xml_request.send(null);
      }

      
      function handle_daemon_action_request(ca_xml_request)
      {
        if ((ca_xml_request.readyState == 4) || (ca_xml_request.readyState == 3))
        {
          var xmlDoc = ca_xml_request.responseXML;
          if (xmlDoc != null)
          {
            var xmlObj=xmlDoc.documentElement;
            var message = xmlObj.getElementsByTagName("message")[0]
  
            document.getElementById("uwl_fpga_command_response").innerHTML = message.childNodes[0].nodeValue;
            control_request();
          }
        }
      }

      function actionFPGA(url)
      {
        var ca_http_request;
        if (window.XMLHttpRequest)
          ca_http_request = new XMLHttpRequest()
        else
          ca_http_request = new ActiveXObject("Microsoft.XMLHTTP");
  
        ca_http_request.onreadystatechange = function() 
        {
          handle_daemon_action_request(ca_http_request)
        }

        ca_http_request.open("GET", url, true)
        ca_http_request.send(null)
      }

      function stopFPGA()
      {
        var url = "?action=true&cmd=stop";
        actionFPGA(url);
      }

      function startFPGA()
      {
        var url = "?action=true&cmd=start";
        actionFPGA(url);
      }

    </script>
<?php
  }


  function printHTML ()
  {
    echo "<h1>UWL FPGA Data Stream Controls</h1>\n";

    echo "<p>These buttons place a request to activate or deactivate the FPGA controls. The FPGA state is queried every ".($this->callback_freq/1000)." seconds</p>\n";

    echo "<center>\n";

    echo "<table class='config' cellspacing='5px' border=1 width='98%'>\n";

    echo   "<tr>\n";
    echo     "<td width='200px'><b>UWL FPGA Request</b></td>\n";
    echo     "<td id='uwl_fpga_request'></td>\n";
    echo   "</tr>\n";

    echo   "<tr>\n";
    echo     "<td width='200px'><b>UWL FPGA State</b></td>\n";
    echo     "<td id='uwl_fpga_state'></td>\n";
    echo   "</tr>\n";

    echo   "<tr>\n";
    echo     "<td width='200px'><b>UWL FPGA Message</b></td>\n";
    echo     "<td id='uwl_fpga_message'></td>\n";
    echo   "</tr>\n";

    echo   "<tr>\n";
    echo     "<td>\n";
    echo       "<input type='button' value='Start' onClick='startFPGA()'/>\n";
    echo       "<input type='button' value='Stop' onClick='stopFPGA()'/>\n";
    echo     "</td>\n";
    echo     "<td id='uwl_fpga_command_response'></td>\n";
    echo   "</tr>\n";

    echo "</table>\n";
  }
  
  function fpgaComm ($command)
  {
    $fpga_socket = new spip_socket();
    if ($fpga_socket->open ($this->fpga_host, $this->fpga_port, 0) == 0)
    {
      #echo "-> REGISTER MEDUSA bpsr_control<br/>\n";
      $fpga_socket->write ("REGISTER MEDUSA bpsr_control\r\n");
      list ($rval, $reply) = $fpga_socket->read();
      #echo "<- ".$rval." ".$reply."<br/>\n";

      list ($rval, $junk) = $fpga_socket->read();
      #echo "<- ".$rval." ".$reply."<br/>\n";

      #echo "-> ".$command."<br/>\n";
      $fpga_socket->write ($command."\r\n");
      list ($rval, $reply) = $fpga_socket->read();
      #echo "<- ".$rval." ".$reply."<br/>\n";

      list ($rval, $junk) = $fpga_socket->read();
      #echo "<- ".$rval." ".$reply."<br/>\n";

      $fpga_socket->shutdown();
      #$fpga_socket->close();

      if ($rval == 0)
      {
        return $reply;
      }
      else
      {
        return "SOCKET ERROR";
      }
    }
    else
    {
      return "OFFLINE";
    }
  }

  function fpgaGetState ()
  {
    $state = "Unknown";
    $message = "Unknown";

    $reply = $this->fpgaComm ("STREAM STATUS");

    $state = rtrim($reply);
    if ($state == "UNCONFIGURED")
      $message = "The hardware requires additional manual steps from expert personnel before streams can be started.";
    else if ($state == "UNKNOWN")
      $message = "The hardware is not connected or accessible.";
    else if ($state == "IGNORED")
      $message = "The hardware is in maintenance/testing mode and intentionally not serving requests.";
    else if ($state == "ERROR")
      $message = "The hardware has experienced an error and requires expert personnel to intervene.";
    else if ($state == "IDLE")
      $message = "The hardware is not streaming data.";
    else if ($state == "ACTIVE")
      $message = "The hardware is streaming data.";
    else if ($state = "OFFLINE")
      $message = "Could not open socket connection to UWB Server ".$this->fpga_host.":".$this->fpga_port;
    else
      $message = "Warning: Unrecognized state [".$state."]";
   
    return array ($state, $message);
  }

  function printUpdateHTML($get)
  {
    # spip_lmc script runs on each client/server, check that it is running and responsive
    $xml  = XML_DEFINITION;
    $xml .= "<uwl_update>";

    list ($state, $message) = $this->fpgaGetState();

    $xml .= "<request>STREAM STATUS</request>"; 
    $xml .= "<state>".$state."</state>"; 
    $xml .= "<message>".$message."</message>"; 

    $xml .= "</uwl_update>";

    header('Content-type: text/xml');
    echo $xml;
  }

  function printActionHTML($get)
  {
    $xml  = XML_DEFINITION;

    $xml .= "<uwl_action>";

    $message = "";

    if (isset($get["cmd"]))
    {
      list ($state, $message) = $this->fpgaGetState();

      $cmd = "";
      if ($get["cmd"] == "start")
      {
        if (($state == "IDLE") || ($state == "IGNORED") || ($state == "UNKNOWN"))
        {
          $reply = $this->fpgaComm ("STREAM START");
          $state = rtrim($reply);
          if ($state == "SUCCESS")
            $message = "Streams started";
          else if ($state == "FAIL")
            $message = "Streams could not be started";
          else if ($state == "BUSY")
            $message = "Not sure what 'BUSY' means!";
          else if ($state == "IGNORED")
            $message = "The hardware is in maintenance/testing mode and intentionally not serving requests.";
          else
            $message = "Warning: Unrecognized state [".$state."]";
        }
        else
        {
          $message = "Hardware state was ".$state.", it must be IDLE to start streams.";
        }
      }
      else if ($get["cmd"] == "stop")
      {
        if (($state == "ACTIVE") || ($state == "IGNORED") || ($state == "UNKNOWN"))
        { 
          $reply = $this->fpgaComm ("STREAM STOP");
          $state = rtrim($reply);
          if ($state == "SUCCESS")
            $message = "Streams stopped";
          else if ($state == "FAIL")
            $message = "Streams could not be stopped";
          else if ($state == "BUSY")
            $message = "Not sure what 'BUSY' means!";
          else if ($state == "IGNORED")
            $message = "The hardware is in maintenance/testing mode and intentionally not serving requests.";
          else
            $message = "Warning: Unrecognized state [".$state."]";
        }
        else
        { 
          $message = "Hardware state was ".$state.", it must be ACTIVE to stop streams.";
        }
      }
      else
        $message = "Error: unrecognized command ".$get["cmd"];
    }
    else
      $message = "Error: No command specified";

    $xml .= "<message>".$message."</message>";

    $xml .= "</uwl_action>";

    header('Content-type: text/xml');
    echo $xml;
  }
}

if ((!isset($_GET["update"])) && (!isset($_GET["action"])))
  $_GET["single"] = "true";
handleDirect("uwl");

