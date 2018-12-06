<?PHP

error_reporting(E_ALL);
ini_set("display_errors", 1);

include_once("../spip.lib.php");
include_once("../spip_webpage.lib.php");
include_once("../spip_socket.lib.php");
include_once("../tcs.lib.php");

class search extends spip_webpage
{

  function search()
  {
    spip_webpage::spip_webpage();

    $this->title = "Search";
    $this->nav_item = "search";

    $this->tcs = new tcs();
    $this->config = spip::get_config();
    $this->streams = array();
    $this->beams = array();

    $this->plot_width = 300;
    $this->plot_height = 225;

    for ($istream=0; $istream<$this->config["NUM_STREAM"]; $istream++)
    {
      list ($host, $ibeam, $subband) = explode (":", $this->config["STREAM_".$istream]);
      list ($freq, $bw, $nchan) = explode (":", $this->config["SUBBAND_CONFIG_".$subband]);
      $beam_name = $this->config["BEAM_".$ibeam];
      $port = $this->config["BEAM_REPACK_SEARCH_PORT"] + $istream;
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
    return $this->tcs->javaScriptCallback()." search_request();";
  }

  function printJavaScriptHead()
  {
    $this->tcs->printJavaScriptHead();
?>
    <script type='text/javascript'>

      function handle_search_request(s_xml_request)
      {
        if (s_xml_request.readyState == 4)
        {
          var xmlDoc = s_xml_request.responseXML;
          if (xmlDoc != null)
          {
            var xmlObj = xmlDoc.documentElement;

            var h, i, j, k;
            var observation;

            var repack_states = xmlObj.getElementsByTagName("repack_search_state");
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
                      url = "/spip/search/index.php?update=true&stream="+stream_id+"&type=plot&plot="+plot_type+"&res=lo&ts="+plot_timestamp;
                      document.getElementById(plot_id).src = url;
                      document.getElementById(plot_ts).value = plot_timestamp;

                      url = "/spip/search/index.php?update=true&stream="+stream_id+"&type=plot&plot="+plot_type+"&res=hi&ts="+plot_timestamp;
                      document.getElementById(plot_link).href= url;

                    }
                  }
                }
              }
            }
          }
        }
      }

      function search_request()
      {
        var url = "?update=true";

        if (window.XMLHttpRequest)
          s_xml_request = new XMLHttpRequest();
        else
          s_xml_request = new ActiveXObject("Microsoft.XMLHTTP");

        s_xml_request.onreadystatechange = function()
        {
          handle_search_request(s_xml_request)
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
    $xml .= "<search_update>";

    $xml_req  = XML_DEFINITION;
    $xml_req .= "<repack_search_request>";
    $xml_req .= "<requestor>search page</requestor>";
    $xml_req .= "<type>state</type>";
    $xml_req .= "</repack_search_request>";

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

    $xml .= "</search_update>";

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

    if (($get["plot"] == "histogram") ||
        ($get["plot"] == "freqtime") ||
        ($get["plot"] == "timeseries"))
    {
      $host      = $this->streams[$istream]["host"];
      $port      = $this->config["BEAM_REPACK_SEARCH_PORT"] + $istream;

      $xml_req  = XML_DEFINITION;
      $xml_req .= "<repack_search_request>";
      $xml_req .= "<requestor>search page</requestor>";
      $xml_req .= "<type>plot</type>";
      $xml_req .= "<plot>".$get["plot"]."</plot>";
      $xml_req .= "<res>".$get["res"]."</res>";
      $xml_req .= "</repack_search_request>";

      $repack_search_socket = new spip_socket();
      $rval = 0;
      $reply = 0;
      if ($repack_search_socket->open ($host, $port, 0) == 0)
      {
        $repack_search_socket->write ($xml_req."\r\n");
        list ($rval, $reply) = $repack_search_socket->read_raw();
      }
      else
      {
        // TODO generate PNG with error text
        echo "ERROR: could not connect to ".$host.":".$port."<BR>\n";
        return;
      }
      $repack_search_socket->close();

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

  function renderObsTable ($beam)
  {
    $cols = 4;
    $fields = array(
      $beam."_source" => "Source",
      $beam."_start" => "UTC_START",
      $beam."_project_id" => "Project ID",
      $beam."_tobs" => "Tobs",
      $beam."_ra" => "RAJ",
      $beam."_observer" => "Observer",
      $beam."_elapsed" => "Elapsed",
      $beam."_dec" => "DECJ"
    );

    echo "<table id='obsTable' width='100%'>\n";

    $keys = array_keys($fields);
    for ($i=0; $i<count($keys); $i++)
    {
      if ($i % $cols == 0)
        echo "  <tr>\n";
      echo "    <th>".$fields[$keys[$i]]."</th>\n";
      echo "    <td><span id='".$keys[$i]."'>--</span></td>\n";
      if (($i+1) % $cols == 0)
        echo "  </tr>\n";
    }
    echo "</table>\n";
  }

  function renderPlotTable ($beam)
  {
    $img_params = "src='/spip/images/blankimage.gif' width='".$this->plot_width."px' height='".$this->plot_height."px'";
    $full_img_params = "src='/spip/images/blankimage.gif' width='800px' height='600px'";

    echo "<table width='100%' id='plotTable'>\n";

    foreach ($this->streams as $istream => $stream)
    {
      $beam = $stream["beam_name"];
      $freq = $stream["freq"];
      $subband = $stream["subband"];
      echo "<tr>\n";

      echo   "<td>Subband ".$subband."<br/>Centre Freq: ".$freq." MHz</td>";

      $prefix = $beam."_".$istream;
      echo   "<td>";
      echo     "<a id='".$prefix."_freqtime_link'>";
      echo       "<img id='".$prefix."_freqtime' ".$img_params."/>";
      echo     "</a>";
      echo     "<input type='hidden' id='".$prefix."_freqtime_ts' value='not set'/>";
      echo   "</td>\n";


      echo   "<td>";
      echo     "<a id='".$prefix."_histogram_link'>";
      echo       "<img id='".$prefix."_histogram' ".$img_params."/>";
      echo     "</a>";
      echo     "<input type='hidden' id='".$prefix."_histogram_ts' value='not set'/>";
      echo   "</td>\n";


      echo   "<td>";
      echo     "<a id='".$prefix."_timeseries_link'>";
      echo       "<img id='".$prefix."_timeseries' ".$img_params."/>";
      echo     "</a>";
      echo     "<input type='hidden' id='".$prefix."_timeseries_ts' value='not set'/>";
      echo   "</td>\n";

      echo "</tr>\n";
    }

    echo "</table>\n";
  }

}
if (!isset($_GET["update"]))
  $_GET["single"] = "true";
handleDirect("search");

