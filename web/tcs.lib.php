<?PHP

error_reporting(E_ALL);
ini_set("display_errors", 1);

include_once("spip.lib.php");
include_once("spip_webpage.lib.php");
include_once("spip_socket.lib.php");

class tcs extends spip_webpage
{
  function tcs ()
  {
    spip_webpage::spip_webpage();

    $this->source_params = array(
      "source" => "Source",
      "ra" => "RAJ",
      "dec" => "DECJ"
    );

    $this->obs_params = array(
      "start" => "UTC_START",
      "project_id" => "Project ID",
      "tobs" => "Tobs",
      "observer" => "Observer",
      "elapsed" => "Elapsed"
    );

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
    return "tcs_request();";
  }

  function printJavaScriptHead()
  {
?>
    <script type='text/javascript'>

      var active_modes = {}
      active_modes["fold"] = "False";
      active_modes["search"] = "False";
      active_modes["continuum"] = "False";

      function get_node_value(node)
      {
        if (node.childNodes.length > 0)
          return node.childNodes[0].nodeValue;
        else
          return "--";
      }

      function handle_tcs_request(t_xml_request)
      {
        if (t_xml_request.readyState == 4)
        {
          var xmlDoc = t_xml_request.responseXML;
          var tcs_utcs = new Array();

          if (xmlDoc != null)
          {
            var xmlObj = xmlDoc.documentElement;

            var h, i, j, k;      
            var source, name, ra, dec;
            var params, observer, project_id, start, elapsed, tobs
            var observation;

            // process the TCS state first
            var tcs_states = xmlObj.getElementsByTagName("tcs_state");
            for (h=0; h<tcs_states.length; h++)
            {
              var tcs_state = tcs_states[h];
              var beams = tcs_state.getElementsByTagName("beam");

              for (i=0; i<beams.length; i++)
              {
                beam = beams[i];
                var beam_name  = beam.getAttribute("name");
                var beam_state = beam.getAttribute("state");

                source = beam.getElementsByTagName("source")[0];
                name   = get_node_value(source.getElementsByTagName("name")[0]);
                ra     = get_node_value(source.getElementsByTagName("ra")[0]);
                dec    = get_node_value(source.getElementsByTagName("dec")[0]);
    
                params   = beam.getElementsByTagName("observation_parameters")[0];
                observer = get_node_value(params.getElementsByTagName("observer")[0]);
                project_id = get_node_value(params.getElementsByTagName("project_id")[0]);
                tobs     = get_node_value(params.getElementsByTagName("expected_length")[0]);

                start    = get_node_value(params.getElementsByTagName("utc_start")[0]);
                elapsed  = get_node_value(params.getElementsByTagName("elapsed_time")[0]);

                document.getElementById(beam_name + "_source").innerHTML = name;
                document.getElementById(beam_name + "_ra").innerHTML = ra;
                document.getElementById(beam_name + "_dec").innerHTML = dec;
                document.getElementById(beam_name + "_observer").innerHTML = observer;
                document.getElementById(beam_name + "_project_id").innerHTML = project_id;
                document.getElementById(beam_name + "_start").innerHTML = start;
                document.getElementById(beam_name + "_elapsed").innerHTML = elapsed;
                document.getElementById(beam_name + "_tobs").innerHTML = tobs;

                modes = beam.getElementsByTagName("processing_modes")[0];

                fold_mode = get_node_value(modes.getElementsByTagName("fold")[0]);
                active_modes["fold"] = fold_mode;
                if (fold_mode == "True")
                  beam_state = beam_state + " Fold"

                search_mode = get_node_value(modes.getElementsByTagName("search")[0]);
                active_modes["search"] = search_mode;
                if (search_mode == "True")
                  beam_state = beam_state + " Search"

                continuum_mode = get_node_value(modes.getElementsByTagName("continuum")[0]);
                active_modes["continuum"] = continuum_mode;;
                if (continuum_mode == "True")
                  beam_state = beam_state + " Continuum"

                document.getElementById(beam_name + "_state").innerHTML = "Beam " + beam_name + ": " + beam_state
                document.getElementById(beam_name + "_tobs").innerHTML = tobs;

                //if (beam_state == "Recording")
                  tcs_utcs[beam_name] = start
                //else
                //  tcs_utcs[beam_name] = ""
              }
            }
          }
        }
      }

      function tcs_request() 
      {
        var url = "/spip/tcs.lib.php?update=true";

        if (window.XMLHttpRequest)
          t_xml_request = new XMLHttpRequest();
        else
          t_xml_request = new ActiveXObject("Microsoft.XMLHTTP");

        t_xml_request.onreadystatechange = function()
        {
          handle_tcs_request(t_xml_request)
        };
        t_xml_request.open("GET", url, true);
        t_xml_request.send(null);
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
      $this->renderObsTable($beam["name"]);
    }
  }

  function printUpdateHTML($get)
  {
    # get all information from TCS too
    $tcses = array();

    if ($this->config["INDEPENDENT_BEAMS"] == "true")
    {
      foreach ($this->beams as $ibeam => $beam)
      {
        array_push ($tcses, $beam["host"].":".($this->config["TCS_REPORT_PORT_".$ibeam]));
      }
    }
    else
    {
      array_push ($tcses, $this->config["SERVER_HOST"].":".$this->config["TCS_REPORT_PORT"]);
    }
  
    $xml_req  = XML_DEFINITION;
    $xml_req .= "<tcs_state_request>";
    $xml_req .= "<requestor>search page</requestor>";
    $xml_req .= "<type>state</type>";
    $xml_req .= "</tcs_state_request>";

    $xml = XML_DEFINITION;

    $xml .= "<tcs_update>";

    foreach ($tcses as $tcs)
    {
      $tcs_socket = new spip_socket();
      list ($host, $port) = explode(":", $tcs);
      if ($tcs_socket->open ($host, $port, 0) == 0)
      {
        $tcs_socket->write ($xml_req."\r\n");
        list ($rval, $reply) = $tcs_socket->read();
        $xml .= rtrim($reply);
        $tcs_socket->close();
      }
      else
      {
        $xml .= "<tcs_state></tcs_state>";
      }
    }

    $xml .= "</tcs_update>";

    header('Content-type: text/xml');
    echo $xml;
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
}
if ($_SERVER["SCRIPT_NAME"] == "/spip/tcs.lib.php")
{
  if (!isset($_GET["update"]))
    $_GET["single"] = "true";
  handleDirect("tcs");
}

