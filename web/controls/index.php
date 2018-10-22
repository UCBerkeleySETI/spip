<?PHP

error_reporting(E_ALL);
ini_set("display_errors", 1);

include_once("../spip_webpage.lib.php");
include_once("../spip.lib.php");
include_once("../spip_socket.lib.php");

class controls extends spip_webpage
{
  var $server_daemons = array();
  var $client_daemons = array();

  function controls ()
  {
    spip_webpage::spip_webpage ();

    $this->title = "Controls";
    $this->nav_item = "controls";

    $this->config = spip::get_config();
    $this->independent_beams = true;

    $this->topology = array();
    $this->server_list = array();
    $this->client_list = array();

    if (strcmp($this->config["INDEPENDENT_BEAMS"], "true") !== TRUE)
    {
      $this->independent_beams = false;
      $host = $this->config["SERVER_HOST"];
      if (!array_key_exists($host, $this->topology))
      {
        $this->topology[$host] = array();
        array_push ($this->server_list, $host);
      }
      array_push ($this->topology[$host], array("beam" => "server", "subband" => "", "stream_id" => "-1"));
    }

    for ($i=0; $i<$this->config["NUM_STREAM"]; $i++)
    {
      list ($host, $beam, $subband) = explode (":", $this->config["STREAM_".$i]);

      if (!array_key_exists($host, $this->topology))
      {
        $this->topology[$host] = array();
        if (!in_array($host, $this->server_list))
        {
          array_push ($this->client_list, $host);
        }
      }
      array_push($this->topology[$host], array("beam" => $beam, "subband" => $subband, "stream_id" => $i));
    }

    // prepare server daemons
    $list = explode (" ", $this->config["SERVER_DAEMONS"]);
    foreach ($list as $item)
    {
      if ($item != "none")
      {
        list ($daemon, $level) = explode (":", $item);
        array_push ($this->server_daemons, array("daemon" => $daemon, "level" => $level));
      }
    }

    // prepare client daemons
    $list = explode (" ", $this->config["CLIENT_DAEMONS"]);
    foreach ($list as $item)
    {
      list ($daemon, $level) = explode (":", $item);
      array_push ($this->client_daemons, array("daemon" => $daemon, "level" => $level));
    }
  }

  function javaScriptCallback()
  {
    return "control_request();";
  }

  function printJavaScriptHead()
  {
?>
    <script type='text/javascript'>

      stop_wait = 0;
      start_wait = 0;

      function examine_daemons (host, id, daemons)
      { 
        for (i=0; i<daemons.length; i++)
        {
          var daemon = daemons[i];
          var daemon_name = daemon.getAttribute("name");
          var daemon_running = daemon.childNodes[0].nodeValue;
          var daemon_id = host + "_" + daemon_name + "_" + id + "_light";
          try {
            var daemon_light = document.getElementById(daemon_id)
            if (daemon_running == "True")
              daemon_light.src = "/spip/images/green_light.png";
            else
              daemon_light.src = "/spip/images/red_light.png";
          } catch (e) {
            alert("ERROR: id=" + daemon_id + " Error=" + e)
          }
        }
      }

      function disable_daemons (host)
      {
        daemons = document.getElementsByTagName("img");
        for (i=0; i<daemons.length; i++)
        {
          var daemon = daemons[i];
          if ((daemon.id != host + "_lmc_light") && (daemon.id.indexOf(host) != -1))
          {
            daemon.src = "/spip/images/grey_light.png";
          }
        }
      }

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
            
            var lmcs = xmlObj.getElementsByTagName("lmc")
            var i, j, k;      
  
            for (i=0; i<lmcs.length; i++)
            {
              var lmc = lmcs[i];

              var host = lmc.getAttribute("host");
              var port = lmc.getAttribute("port");

              var state = lmc.getElementsByTagName("state")[0];
              var lmc_light = document.getElementById(host + "_lmc_light")
              if (state.childNodes[0].nodeValue == "Running")
              {
                lmc_light.src = "/spip/images/green_light.png";

                var servers = lmc.getElementsByTagName("server")
                for (j=0; j<servers.length; j++)
                {
                  var server = servers[j];
                  var server_id = server.getAttribute("id")
                  daemons = server.getElementsByTagName("daemon");
                  examine_daemons (host, server_id, daemons)
                }

                var streams = lmc.getElementsByTagName("stream")
                for (j=0; j<streams.length; j++)
                {
                  var stream = streams[j];
                  var stream_id = stream.getAttribute("id")
                  daemons = stream.getElementsByTagName("daemon");
                  examine_daemons (host, stream_id, daemons)
                }
              }
              else
              {
                lmc_light.src = "/spip/images/red_light.png";
                disable_daemons (host);
              }
            }
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
          }
        }
      }

      function actionLMC(url)
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

      function stopLMC(host, stream)
      {
        var url = "?action=true&host="+host+"&cmd=stop_lmc&stream="+stream;
        actionLMC(url);
      }

      function startLMC(host, stream)
      {
        var url = "?action=true&host="+host+"&cmd=start_lmc&stream="+stream;
        actionLMC(url);
      }

      function controlMessage (id, message)
      {
        document.getElementById(id).innerHTML = message;
      }

      function startAll()
      {
        controlMessage("stop_all_reply", "");
        controlMessage("start_all_reply", "Starting Medusa services");
        disableAll();
<?php
        foreach ($this->server_list as $host)
          echo "        startLMC ('".$host."', 'server');\n";
        foreach ($this->client_list as $host)
          echo "        startLMC ('".$host."', '0');\n";
?>
        start_wait = 60;
        waitForStart();
      }

      function stopAll()
      {
        controlMessage("start_all_reply", "");
        controlMessage("stop_all_reply", "Stopping Medusa services");
        disableAll();
<?php
        foreach ($this->client_list as $host)
          echo "        stopLMC ('".$host."', '0');\n";
        foreach ($this->server_list as $host)
          echo "        stopLMC ('".$host."', 'server');\n";
?>
        stop_wait = 60;
        waitForStop();
      }

      function disableAll()
      {
        document.getElementById("start_all").disabled = true;
        document.getElementById("stop_all").disabled = true;
<?php
        foreach ($this->client_list as $host)
        {
          echo "        document.getElementById('start_".$host."').disabled = true;\n";
          echo "        document.getElementById('stop_".$host."').disabled = true;\n";
        }
        foreach ($this->server_list as $host)
        {
          echo "        document.getElementById('start_".$host."').disabled = true;\n";
          echo "        document.getElementById('stop_".$host."').disabled = true;\n";
        }
?>
      }

      function enableAll()
      {
        document.getElementById("start_all").disabled = false;
        document.getElementById("stop_all").disabled = false;
<?php   
        foreach ($this->client_list as $host)
        {
          echo "        document.getElementById('start_".$host."').disabled = false;\n";
          echo "        document.getElementById('stop_".$host."').disabled = false;\n";
        }
        foreach ($this->server_list as $host)
        {
          echo "        document.getElementById('start_".$host."').disabled = false;\n";
          echo "        document.getElementById('stop_".$host."').disabled = false;\n";
        }
?>
      }

      function checkStates(colours)
      {
        var i=0;
        var j=0;
        var ready = true;
        elements = document.getElementsByTagName("img")
        for (i=0; i<elements.length; i++)
        {
          element_ready = false;
          for (j=0; j<colours.length; j++)
          {
            if (elements[i].src.indexOf(colours[j]) != -1)
              element_ready = true
          }
          if (!element_ready)
            ready = false;
        }
        return ready;
      }

      function waitForStop()
      {
        colors = new Array ("red", "grey");
        if (!checkStates(colors))
        {
          stop_wait--;
          controlMessage("stop_all_reply", "Stopping Medusa services [timeout " + stop_wait + "]");
          setTimeout ('waitForStop()', 1000);
          return 0;
        }
        stop_wait = 0;
        controlMessage("stop_all_reply", "All Medusa services stopped");
        enableAll();
      }

      function waitForStart()
      {
        colors = new Array ("green");
        if (!checkStates(colors))
        {
          start_wait--;
          controlMessage ("start_all_reply", "Starting Medusa services [timeout " + start_wait + "]");
          setTimeout ('waitForStart()', 1000);
          return 0;
        }
        start_wait = 0;
        controlMessage("start_all_reply", "All Medusa services started");
        enableAll();
      }

    </script>
<?php
  }


  function printHTML ()
  {
    echo "<h1>Controls</h1>\n";

    echo "<h2>Instrument Controls</h2>\n";

    echo "<p>Use these buttons to start or stop the Medusa Backend in the correct sequence.\n";
    echo "Note that it may take 10s of seconds for the startup or shutdown to complete.</p>\n";

    echo "<table cellpadding='5px'>\n";

    echo "<tr>\n";
    echo  "<td><b>Control</b></td>\n";
    echo  "<td><b>Response</b></td>\n";
    echo "</tr>\n";

    echo "<tr>\n";
    echo  "<td>";
    echo    "<input id='start_all' type='button' onClick='startAll()' value='Start Instrument'/><br/>";
    echo  "</td>";
    echo  "<td>";
    echo    "<div id='start_all_reply'></div>\n";
    echo  "</td>\n";
    echo  "</tr>\n";
      
    echo "<tr>\n";
    echo  "<td>";
    echo    "<input id='stop_all' type='button' onClick='stopAll()' value='Stop Instrument'/>\n";
    echo  "</td>\n";
    echo  "<td>";
    echo    "<div id='stop_all_reply'></div>\n";
    echo  "</td>\n";
    echo  "</tr>\n";

    echo "</table>\n";

    echo "<p>When starting Medusa, all the Individual Server Controls must be OFF (red or grey). Click the Start Instrument button once and wait for the all the controls to change from OFF (red/grey) to ON (green).</p>\n";

    echo "<p>When stopping Medusa, all the Individual Server Controls must be on (green). Click the Stop Instrument button once and wait for all the controls to change from ON (green) to OFF (red/grey).</p>\n";

    echo "<hr>\n";

    echo "<h2>Individual Server Controls</h2>\n";

    echo "<center>\n";

    # if the beams are to be operated independently of each other, then separate
    # server and client daemons will exist for each beam. However, if the beams
    # are used together (i.e. for a multi-beam survey), only 1 set of server daemons
    # will be used

    echo "<table class='config' border=1 width='98%'>\n";

    echo "<tr>\n";
    echo "<th>Host</th>\n";
    echo "<th>LMC</th>\n";
    if ($this->config["NUM_BEAM"] > 1)
      echo "<th>Beam</th>\n";
    if ($this->config["NUM_SUBBAND"] > 1)
      echo "<th>Sub-band</th>\n";
    echo "<th>Daemons</th>\n";
    echo "</tr>\n";

    $hosts = array_keys($this->topology);

    foreach ($hosts as $host)
    {
      $host_rows = count($this->topology[$host]);
      for ($i=0; $i<count($this->topology[$host]); $i++)
      {
        $stream = $this->topology[$host][$i];

        echo "<tr>\n";

        if ($host_rows == 1 || ($host_rows > 1 && $i == 0))
        {
          echo "<td rowspan=".$host_rows.">".$host."</td>\n";
          # each host has a single LMC instance that manages all child daemons
          echo "<td rowspan=".$host_rows." width='150px'>\n";
            echo "<img border='0' id='".$host."_lmc_light' src='/spip/images/grey_light.png' width='15px' height='15px'>\n";
            echo "<input id='start_".$host."' type='button' value='Start' onClick='startLMC(\"".$host."\",\"".$stream["beam"]."\")'/>\n";
            echo "<input id='stop_".$host."' type='button' value='Stop' onClick='stopLMC(\"".$host."\",\"".$stream["beam"]."\")'/>\n";
          echo "</td>\n";
        }

        if ($this->config["NUM_BEAM"] > 1)
          echo "<td>".$stream["beam"]."</td>\n";
        if ($this->config["NUM_SUBBAND"] > 1)
          echo "<td>".$stream["subband"]."</td>\n";

        echo "<td>\n";
        if ($stream["beam"] == "server")
        {
          foreach ($this->server_daemons as $d)
          {
            $id = $host."_".$d["daemon"]."_".$stream["stream_id"];
            echo "<span style='padding-right: 10px;'>\n";
            echo "<img border='0' id='".$id."_light' src='/spip/images/grey_light.png' width='15px' height='15px'>\n";
            echo $d["daemon"];
            echo "</span>\n";
          }
        }
        else
        {
          foreach ($this->client_daemons as $d)
          {
            echo "<span style='padding-right: 10px;'>\n";
            $id = $host."_".$d["daemon"]."_".$stream["stream_id"];
            echo "<img border='0' id='".$id."_light' src='/spip/images/grey_light.png' width='15px' height='15px'>&nbsp;";
            echo $d["daemon"];
            echo "</span>\n";
          }
        }
        echo "</td>\n";
       
        echo "</tr>\n";
      }
    }

/*
    echo "<tr>\n";

    echo "<td colspan=2>";
    echo "<td>\n";
    echo "<input type='button' value='Start All' onClick='startServer(\"all\",\"all\")'/>\n";
    echo "<input type='button' value='Stop All' onClick='stopServer(\"all\",\"all\")'/>\n";
    echo "</td>\n";
    echo "<td>\n";
    echo "<input type='button' value='Start All' onClick='startClient(\"all\",\"all\")'/>\n";
    echo "<input type='button' value='Stop All' onClick='stopClient(\"all\",\"all\")'/>\n";
    echo "</td>\n";

    echo "</tr>\n";
*/

    echo "</table>\n";
  }
  
  function printUpdateHTML($get)
  {
    # echo "control::index::printUpdateHTML<BR>\n";
    # spip_lmc script runs on each client/server, check that it is running and responsive
    $xml = "<controls_update>";

    # check if the LMC script is running on the specified host
    $hosts = array_keys($this->topology);
    $port  = $this->config["LMC_PORT"];

    $xml_cmd  = XML_DEFINITION;
    $xml_cmd .= "<lmc_cmd>";
    $xml_cmd .= "<requestor>controls page</requestor>";
    $xml_cmd .= "<command>daemon_status</command>";
    $xml_cmd .= "</lmc_cmd>";

    foreach ($hosts as $host)
    {
      $xml .= "<lmc host='".$host."' port='".$port."'>";
      $lmc_socket = new spip_socket();
      $connected = $lmc_socket->open ($host, $port, 0);
      if ($connected == 0)
      {
        $xml .= "<state>Running</state>";
        $lmc_socket->write ($xml_cmd."\r\n");
        list ($rval, $reply) = $lmc_socket->read();
        if ($rval == 0)
          $xml .= $reply;
        $lmc_socket->close();
      }
      else
      {
        $xml .= "<state>Offline</state>";
      }
      $xml .= "</lmc>";
    }

    $xml .= "</controls_update>";

    header('Content-type: text/xml');
    echo $xml;
  }

  function printActionHTML($get)
  {
    $xml = "<controls_action>";

    if (isset($get["cmd"]) && (($get["cmd"] == "start_lmc") || ($get["cmd"] == "stop_lmc")))
    {
      $cmd = "";
      if ($get["cmd"] == "start_lmc")
      {
        $cmd = "ssh ".$get["host"]." '".$this->config["SCRIPTS_DIR"]."/spip.init start'";
      }
      else if ($get["cmd"] == "stop_lmc")
      {
        $cmd = "ssh ".$get["host"]." '".$this->config["SCRIPTS_DIR"]."/spip.init stop'"; 
      }
      else
      {
        $xml .= "<message>Error: unrecognized command ".$get["cmd"]."</message>";
      }
      $xml .= "<cmd>".$cmd."</cmd>";
      if ($cmd != "")
      {
        $lines = array();
        $last = exec($cmd, $lines, $rval);
        if ($rval == 0)
        {
          $xml .= "<message>ok</message>";
        }
        else
        {
          $html_lines = join($lines, "<BR>");
          $xml .= "<message rval='".$rval."'>".$html_lines."</message>";
        }
      }
    }
    else
    {
      $xml .= "<message>Error: no cmd specified</message>";
    }

    $xml .= "</controls_action>";

    header('Content-type: text/xml');
    echo $xml;
  } 
}

if ((!isset($_GET["update"])) && (!isset($_GET["action"])))
  $_GET["single"] = "true";
handleDirect("controls");

