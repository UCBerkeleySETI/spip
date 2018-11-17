<?PHP

error_reporting(E_ALL);
ini_set("display_errors", 1);

include_once("../spip.lib.php");
include_once("../spip_webpage.lib.php");
include_once("../spip_socket.lib.php");
include_once("../tcs.lib.php");

class preproc extends spip_webpage
{

  function preproc()
  {
    spip_webpage::spip_webpage();

    $this->title = "preproc";
    $this->nav_item = "preproc";

    $this->tcs = new tcs();
    $this->config = spip::get_config();
    $this->streams = array();
    $this->beams = array();

    $this->plot_width = 240;
    $this->plot_height = 180;

    for ($istream=0; $istream<$this->config["NUM_STREAM"]; $istream++)
    {
      list ($host, $ibeam, $subband) = explode (":", $this->config["STREAM_".$istream]);
      list ($freq, $bw, $nchan) = explode (":", $this->config["SUBBAND_CONFIG_".$subband]);
      $beam_name = $this->config["BEAM_".$ibeam];
      $port = $this->config["STREAM_PREPROC_PORT"] + $istream;
      $this->streams[$istream] = array("beam_name" => $beam_name, "host" => $host, "port" => $port, "subband" => $subband, "freq" => $freq);
    }

    for ($ibeam=0; $ibeam<$this->config["NUM_BEAM"]; $ibeam++)
    {
      $beam_name = $this->config["BEAM_".$ibeam];
      $primary_subband = $this->config["NUM_SUBBAND"];
      $primary_stream = -1;

      # find the lowest indexed stream for this beam
      for ($istream=0; $istream<$this->config["NUM_STREAM"]; $istream++)
      {
        list ($host, $beam, $subband) = explode (":", $this->config["STREAM_".$istream]);
        if (($beam == $ibeam) && ($subband < $primary_subband))
        {
          $primary_subband = $subband;
          $primary_stream = $istream;
        }
      }

      list ($host, $beam, $subband) = explode (":", $this->config["STREAM_".$primary_stream]);
      $this->beams[$ibeam] = array ("name" => $beam_name, "host" => $host);
    }
  }

  function javaScriptCallback()
  {
    return $this->tcs->javaScriptCallback()." preproc_request();";
  }

  function printJavaScriptHead()
  {
    $this->tcs->printJavaScriptHead();
?>
    <script type='text/javascript'>

      function handle_preproc_request(s_xml_request)
      {
        if (s_xml_request.readyState == 4)
        {
          var xmlDoc = s_xml_request.responseXML;
          var tcs_utcs = new Array();

          if (xmlDoc != null)
          {
            var xmlObj = xmlDoc.documentElement;

            var h, i, j, k;      
            var observation;

            var repack_states = xmlObj.getElementsByTagName("preproc_state");
            for (h=0; h<repack_states.length; h++)
            {
              var repack_state = repack_states[h];
              var streams = repack_state.getElementsByTagName("stream");
              for (i=0; i<streams.length; i++)
              {
                var stream = streams[i];

                var stream_id = stream.getAttribute("id");
                var beam_name = stream.getAttribute("beam_name");
                var active    = stream.getAttribute("active");

                var plots = Array();

                //if (active == "True")
                {
                  observation = stream.getElementsByTagName("observation")[0]
                  plots = stream.getElementsByTagName("plot");

                  for (j=0; j<plots.length; j++)
                  {
                    var plot = plots[j]
                    var plot_type = plots[j].getAttribute("type")  
                    var plot_timestamp = plots[j].getAttribute("timestamp")  
    
                    var plot_id = beam_name + "_" + stream_id + "_" + plot_type
                    var plot_ts = beam_name + "_" + stream_id + "_" + plot_type + "_ts"
                    var plot_link = beam_name + "_" + stream_id + "_" + plot_type + "_link"

                    // if the image has been updated, reacquire it
                    //alert (plot_timestamp + " ?=? " + document.getElementById(plot_ts).value)
                    if (plot_timestamp != document.getElementById(plot_ts).value)
                    {
                      url = "/spip/preproc/index.php?update=true&stream="+stream_id+"&type=plot&plot="+plot_type+"&res=lo&ts="+plot_timestamp;
                      document.getElementById(plot_id).src = url;
                      document.getElementById(plot_ts).value = plot_timestamp;

                      url = "/spip/preproc/index.php?update=true&stream="+stream_id+"&type=plot&plot="+plot_type+"&res=hi&ts="+plot_timestamp;
                      document.getElementById(plot_link).href= url;

                    }
                  }
                }
              }
            }
          }
        }
      }

      function preproc_request() 
      {
        var url = "?update=true";

        if (window.XMLHttpRequest)
          s_xml_request = new XMLHttpRequest();
        else
          s_xml_request = new ActiveXObject("Microsoft.XMLHTTP");

        s_xml_request.onreadystatechange = function()
        {
          handle_preproc_request(s_xml_request)
        };
        s_xml_request.open("GET", url, true);
        s_xml_request.send(null);
      }

    </script>

    <style type='text/css'>
      #obsTable table {
        border: 1;
      }

      #obsTable th {
        text-align: right;
        padding-right: 5px;
      }

      #obsTable td {
        text-align: left;
      }

      #plotTable table {
        border: 0;
      }

      #plotTable td {
        text-align: center;
      }

    </style>
<?php
  }

  function printHTML()
  {
    foreach ($this->beams as $ibeam => $beam)
    {
      echo "<h2 id='".$beam["name"]."_state'>Beam ".$beam["name"]."</h2>\n";

      $this->tcs->renderObsTable($beam["name"]);
      $this->renderPlotTable($beam["name"]);
    }
  }

  function printUpdateHTML($get)
  {
    if (isset($get["plot"]))
    {
      $this->renderImage($get);
      return;
    }
    
    $xml = XML_DEFINITION;
    $xml .= "<preproc_update>";

    $xml_req  = XML_DEFINITION;
    $xml_req .= "<preproc_request>";
    $xml_req .= "<requestor>preproc page</requestor>";
    $xml_req .= "<type>state</type>";
    $xml_req .= "</preproc_request>";

    foreach ($this->streams as $istream => $stream)
    {
      $repack_socket = new spip_socket();
      $host = $stream["host"];
      $port = $stream["port"];
            
      if ($repack_socket->open ($host, $port, 0) == 0)
      {
        $repack_socket->write ($xml_req."\r\n");
        list ($rval, $reply) = $repack_socket->read();
        $xml .= rtrim($reply);
        $repack_socket->close();
      }
      else
      {
        $xml .= "<repack_state><stream id='".$istream."' active='False' host='".$host."' port='".$port."'></stream></repack_state>";
      }
    }

    $xml .= "</preproc_update>";

    header('Content-type: text/xml');
    echo $xml;
  }

  // will contact a repacker to request current image information
  function renderImage($get)
  {
    $istream = $get["stream"];

    if ($istream < 0 || $istream > $this->config["NUM_STREAM"])
    {
      echo "ERROR: could not identify stream<br/>\n";
      return;
    }

    if (($get["plot"] == "gainstime") || 
        ($get["plot"] == "gainsfreq") || 
        ($get["plot"] == "dirty") || 
        ($get["plot"] == "cleaned"))
    {
      $host      = $this->streams[$istream]["host"];
      $port      = $this->config["STREAM_PREPROC_PORT"] + $istream;

      $xml_req  = XML_DEFINITION;
      $xml_req .= "<preproc_request>";
      $xml_req .= "<requestor>preproc page</requestor>";
      $xml_req .= "<type>plot</type>";
      $xml_req .= "<plot>".$get["plot"]."</plot>";
      $xml_req .= "<res>".$get["res"]."</res>";
      $xml_req .= "</preproc_request>";

      $repack_preproc_socket = new spip_socket();
      $rval = 0;
      $reply = 0;
      if ($repack_preproc_socket->open ($host, $port, 0) == 0)
      {
        $repack_preproc_socket->write ($xml_req."\r\n");
        list ($rval, $reply) = $repack_preproc_socket->read_raw();
      }
      else
      {
        // TODO generate PNG with error text
        echo "ERROR: could not connect to ".$host.":".$port."<BR>\n";
        return;
      }
      $repack_preproc_socket->close();

      if ($rval == 0)
      {
        header('Content-type: image/png');
        header('Content-Disposition: inline; filename="image.png"');
        echo $reply;
      }
    }
    else
    {
      ;
    }
  }

  function renderPlotTable ($beam)
  {
    $img_params = "src='/spip/images/blankimage.gif' width='".$this->plot_width."px' height='".$this->plot_height."px'";
    $full_img_params = "src='/spip/images/blankimage.gif' width='800px' height='600px'";

    echo "<h1>Adaptive Filter</h1>\n";
    echo "<table width='100%' id='plotTable'>\n";
    echo "<tr>\n";
    echo   "<td>Freq. [MHz]</td>\n";
    echo   "<td>Max Gain v Time</td>\n";
    echo   "<td>Curr. Gain v Freq</td>\n";
    echo   "<td>Input Bandpass</td>\n";
    echo   "<td>Output Bandpass</td>\n";

    foreach ($this->streams as $istream => $stream)
    {
      $beam = $stream["beam_name"];
      $freq = $stream["freq"];
      $subband = $stream["subband"];
      echo "<tr>\n";

      echo   "<td>".$freq."</td>";

      $prefix = $beam."_".$istream;
      echo   "<td>";
      echo     "<a id='".$prefix."_gainstime_link'>";
      echo       "<img id='".$prefix."_gainstime' ".$img_params."/>";
      echo     "</a>";
      echo     "<input type='hidden' id='".$prefix."_gainstime_ts' value='not set'/>";
      echo   "</td>\n";

      echo   "<td>";
      echo     "<a id='".$prefix."_gainsfreq_link'>";
      echo       "<img id='".$prefix."_gainsfreq' ".$img_params."/>";
      echo     "</a>";
      echo     "<input type='hidden' id='".$prefix."_gainsfreq_ts' value='not set'/>";
      echo   "</td>\n";

      echo   "<td>";
      echo     "<a id='".$prefix."_dirty_link'>";
      echo       "<img id='".$prefix."_dirty' ".$img_params."/>";
      echo     "</a>";
      echo     "<input type='hidden' id='".$prefix."_dirty_ts' value='not set'/>";
      echo   "</td>\n";

      echo   "<td>";
      echo     "<a id='".$prefix."_cleaned_link'>";
      echo       "<img id='".$prefix."_cleaned' ".$img_params."/>";
      echo     "</a>";
      echo     "<input type='hidden' id='".$prefix."_cleaned_ts' value='not set'/>";
      echo   "</td>\n";

      echo "</tr>\n";
    }

    echo "</table>\n";

    echo "<h1>System Temperature</h1>\n";

    echo "<table width='100%' id='plotTable'>\n";
    echo "<tr>\n";
    echo   "<td>Freq. [MHz]</td>\n";
    echo   "<td>TSYS</td>\n";
    echo "</tr>\n";

    $img_params = "src='/spip/images/blankimage.gif' width='900px' height='".$this->plot_height."px'";
    $full_img_params = "src='/spip/images/blankimage.gif' width='800px' height='600px'";

    foreach ($this->streams as $istream => $stream)
    {
      $beam = $stream["beam_name"];
      $freq = $stream["freq"];
      $subband = $stream["subband"];
      echo "<tr>\n";

      echo   "<td>".$freq."</td>";

      $prefix = $beam."_".$istream;

      echo   "<td>";
      echo     "<a id='".$prefix."_tsys_link'>";
      echo       "<img id='".$prefix."_tsys' ".$img_params."/>";
      echo     "</a>";
      echo     "<input type='hidden' id='".$prefix."_tsys_ts' value='not set'/>";
      echo   "</td>\n";

      echo "</tr>\n";
    }

    echo "</table>\n";


  }
    
}
if (!isset($_GET["update"]))
  $_GET["single"] = "true";
handleDirect("preproc");

