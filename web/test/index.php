<?PHP

ini_set("display_errors", "on");
error_reporting(E_ALL);

include_once("../spip_webpage.lib.php");
include_once("../spip.lib.php");
include_once("../spip_socket.lib.php");

class tests extends spip_webpage
{
  function tests ()
  {
    spip_webpage::spip_webpage ();

    $this->title = "Test System";
    $this->nav_item = "test";

    # configuration definition for the observation
    $this->source_config = $this->get_source_config();
    $this->obs_config = $this->get_obs_config();
    $this->beam_config = $this->get_beam_config();
    $this->custom_config = $this->get_custom_config();
    $this->proc_modes = $this->get_proc_modes();
    $this->fold_config = $this->get_fold_config();
    $this->search_config = $this->get_search_config();
    $this->continuum_config = $this->get_continuum_config();
    $this->baseband_config = $this->get_baseband_config();
    $this->vlbi_config = $this->get_vlbi_config();
  }
  
  function build_hash ($prefix, $xml, $key, $name, $type, $value, $size, $attr="", $onchange="")
  {
    return array ("prefix" => $prefix, "tag" => $xml, "key" => $key, "name" => $name, 
                  "type" => $type, "value" => $value, "attr" => $attr, "size" => $size, 
                  "maxlength" => "", "onchange" => $onchange);
  }

  function get_source_config()
  {
    $a = array();
    array_push($a, $this->build_hash("source", "name", "SOURCE", "Source Name", "text", "J0437-4715", "16", "epoch='J2000'"));
    array_push($a, $this->build_hash("source", "ra", "RA", "Right Ascension", "text", "04:37", "16", "units='hh:mm:ss'"));
    array_push($a, $this->build_hash("source", "dec", "DEC", "Declination", "text", "47:15", "16", "units='dd:mm:ss'"));
    return $a;
  }

  function get_obs_config()
  {
    $a = array();
    array_push($a, $this->build_hash("obs", "observer", "OBSERVER", "Observer", "text", "Andrew", "16"));
    array_push($a, $this->build_hash("obs", "project_id", "PID", "Project ID", "text", "P999", "8"));
    array_push($a, $this->build_hash("obs", "tobs", "TOBS", "Expected Length [s]", "text", "60", "8"));
    array_push($a, $this->build_hash("obs", "calfreq", "CALFREQ", "Calibrator Frequency [Hz]", "text", "11.123", "8"));
    return $a;
  }

  function get_beam_config()
  {
    $nbeam = $this->config["NUM_BEAM"];
    $a = array();
    array_push($a, $this->build_hash("beam", "nbeam", "NBEAM", "Number of beams", "text", $nbeam, "4"));
    for ($i=0; $i<$nbeam; $i++)
    {
      array_push($a, $this->build_hash("beam", "beam_state_".$i, "BEAM_STATE_".$i,  "Beam ".$i, "bool", ($i == 0), "4", "name='".$this->config["BEAM_".$i]."'"));
    }
    return $a;
  }

  function get_custom_config()
  {
    $a = array();
    array_push($a, $this->build_hash("custom", "adaptive_filter_epsilon", "ADAPTIVE_FILTER_EPSILON", "Adaptive Filter Epsilon", "text", "0.1", "4"));
    array_push($a, $this->build_hash("custom", "adaptive_filter_nchan", "ADAPTIVE_FILTER_NCHAN", "Adaptive Filter Channels", "text", "128", "5"));
    array_push($a, $this->build_hash("custom", "adaptive_filter_nsamp", "ADAPTIVE_FILTER_NSAMP", "Adaptive Filter Samples", "text", "1024", "5"));
    return $a;
  }

  function get_proc_modes()
  {
    $a = array();
    array_push($a, $this->build_hash("proc", "fold", "PERFORM_FOLD", "Fold Mode", "bool", "true", "2", "", "showProcessingMode('fold')"));
    array_push($a, $this->build_hash("proc", "search", "PERFORM_SEARCH", "Search Mode", "bool", "false", "2", "", "showProcessingMode('search')"));
    array_push($a, $this->build_hash("proc", "continuum", "PERFORM_CONTINUUM", "Continuum Mode", "bool", "false", "2", "", "showProcessingMode('continuum')"));
    array_push($a, $this->build_hash("proc", "spectral_line", "PERFORM_SPECTRAL_LINE", "Spectral Line Mode", "bool", "false", "2", "", "showProcessingMode('spectral_line')"));
    array_push($a, $this->build_hash("proc", "vlbi", "PERFORM_VLBI", "VLBI Mode", "bool", "false", "2", "", "showProcessingMode('vlbi')"));
    array_push($a, $this->build_hash("proc", "baseband", "PERFORM_BASEBAND", "Baseband Mode", "bool", "false", "2", "", "showProcessingMode('baseband')"));
    return $a;
  }

  function get_fold_config()
  {
    $a = array();
    array_push($a, $this->build_hash("fold", "output_nchan", "FOLD_OUTNCHAN", "Number of output channels", "text", "128", "8"));
    array_push($a, $this->build_hash("fold", "custom_dm", "FOLD_DM", "Custom DM", "text", "-1", "8"));
    array_push($a, $this->build_hash("fold", "output_nbin", "FOLD_OUTNBIN", "Number of output phase bins", "text", "1024", "8"));
    array_push($a, $this->build_hash("fold", "output_tsubint", "FOLD_OUTTSUBINT", "Output subint length [s]", "text", "10", "8"));
    array_push($a, $this->build_hash("fold", "output_npol", "FOLD_OUTNPOL", "Number of output polarisations", "text", "4", "1"));
    array_push($a, $this->build_hash("fold", "mode", "MODE", "Observing Type", "radio", array("PSR" => "true", "CAL" => "false"), "8"));
    array_push($a, $this->build_hash("fold", "sk", "FOLD_SK", "Spectral Kurtosis", "bool", "false", "8"));
    array_push($a, $this->build_hash("fold", "sk_threshold", "FOLD_SK_THRESHOLD", "Spectral Kurtosis Threshold", "text", "3", "8"));
    array_push($a, $this->build_hash("fold", "sk_nsamps", "FOLD_SK_NSAMPS", "Spectral Kurtosis Samples", "text", "1024", "8"));
    return $a;
  }

  function get_search_config()
  {
    $a = array();
    array_push($a, $this->build_hash("search", "output_nchan", "SEARCH_OUTNCHAN", "Output channels", "text", "1024", "8"));
    array_push($a, $this->build_hash("search", "custom_dm", "SEARCH_DM", "Custom DM", "text", "-1", "8"));
    array_push($a, $this->build_hash("search", "output_nbit", "SEARCH_OUTNBIT", "Output bits per sample", "text", "8", "2"));
    array_push($a, $this->build_hash("search", "output_tdec", "SEARCH_OUTTDEC", "Sample integration factor", "text", "512", "8"));
    array_push($a, $this->build_hash("search", "output_tsubint", "SEARCH_OUTTSUBINT", "Output subint length [s]", "text", "10", "8"));
    array_push($a, $this->build_hash("search", "output_npol", "SEARCH_OUTNPOL", "Number of output polarisations", "text", "4", "1"));
    return $a;
  }

  function get_continuum_config()
  {
    $a = array();
    array_push($a, $this->build_hash("continuum", "output_nchan", "CONTINUUM_OUTNCHAN", "Output channels", "text", "32768", "8"));
    array_push($a, $this->build_hash("continuum", "output_tsubint", "CONTINUUM_OUTTSUBINT", "Output integration length [s]", "text", "10", "8"));
    array_push($a, $this->build_hash("continuum", "output_npol", "CONTINUUM_OUTNPOL", "Number of output polarisations", "text", "4", "1"));
    return $a;
  }

  function get_vlbi_config()
  {
    $a = array();
    array_push($a, $this->build_hash("vlbi", "auto_gain", "VLBI_AUTO_GAIN", "Automatic gain control", "bool", "true", "8"));
    array_push($a, $this->build_hash("vlbi", "level_setting", "VLBI_LEVEL_SETTING", "Level setting Options", "select", array("1" => "No Level Setting", "1" => "Adaptive", "2" => "Constant"), "8"));
    array_push($a, $this->build_hash("vlbi", "level_time_scale", "VLBI_TIME_SCALE", "Level setting timescale", "text", 1, "8"));
    array_push($a, $this->build_hash("vlbi", "output_nbit", "OUTNBIT", "Output bits per sample", "text", "8", "2"));
    array_push($a, $this->build_hash("vlbi", "output_bw", "OUTBW", "Output bandwidth", "select", array("16" => "16 MHz", "32" => "32 MHz", "64" => "64 MHz", "128" => "128 MHz"), "8"));
    return $a;
  }


  function get_baseband_config()
  {
    $a = array();
    array_push($a, $this->build_hash("baseband", "output_nbit", "OUTNBIT", "Output bits per sample", "text", "16", "2"));
    return $a;
  }


  function printJavaScriptHead()
  {
?>
    <script type='text/javascript'>

      function configureButton()
      {
        document.getElementById("command").value = "configure";
        document.tcs.submit();
      }

      function startButton()
      {
        document.getElementById("command").value = "start";
        document.tcs.submit();
      }

      function stopButton() {
        document.getElementById("command").value = "stop";
        document.tcs.submit();
      }

      function showProcessingMode(mode)
      {
        var modes = Array("fold", "search", "continuum", "spectral_line", "vlbi", "baseband");
        for (i=0; i<modes.length; i++)
        {
          if (mode == modes[i])
          {
            document.getElementById(modes[i]).style.display = "table";
          }
          else
          {
            document.getElementById(modes[i]).style.display = "none";
            document.getElementById("proc_" + modes[i]).checked = false;
          }
        }
      }
    </script>
<?php
  } 

  function printHTML ()
  {
?>
<h1>Observing Parameters</h1>

<center>

<form name="tcs" target="spip_response" method="GET">

<input type="hidden" name="command" id="command" value=""/>
<table cellpadding='3px' border=0 cellspacing=20px width='100%'>

<tr>
  <td valign=top>

<?php

  $this->renderSourceConfig();
  $this->renderObsConfig();
  $this->renderCustomConfig();
  $this->renderBeamConfig();
?>

  </td>
  <td valign=top>

<?php
  $this->renderProcModes();
  $this->renderFoldMode();
  $this->renderSearchMode();
  $this->renderContinuumMode();
  $this->renderSpectralLineMode();
  $this->renderVLBIMode();
  $this->renderBasebandMode();

/*
  if ($this->config["NUM_BEAM"] == "1")
  {
    echo "<input type='hidden' name='beam_state_0' id='beam_state_0' value=on'>\n";
  }
  else
  {
    for ($i=0; $i<$this->config["NUM_BEAM"]; $i++)
    {
      echo "<tr>";
      echo   "<td>BEAM ".$this->config["BEAM_".$i]."</td>";
      echo   "<td>";
      echo     "<input type='radio' name='beam_state_".$i."' id='beam_state_".$i."' value='on'/><label for='beam_state_".$i."'>On</label>";
      echo     "&nbsp;&nbsp;";
      echo     "<input type='radio' name='beam_state_".$i."' id='beam_state_".$i."' value='off' checked/><label for='beam_state_".$i."'>Off</label>";
      echo   "</td>";
      echo "</tr>\n";
    }
  }
*/

?>
    </td>

    <td valign=top>

    <table class='config' width='100%'>
      <tr>
        <th colspan=2>Instrument Configuration</th>
      </tr>
<?php
      $this->renderInstrumentRow("NUM_BEAM", "NBEAM", "Number of Beams", 1);
      $this->renderInstrumentRow("NBIT", "NBIT", "Bits per sample", 2);
      $this->renderInstrumentRow("NDIM", "NDIM", "Dimensions (complex / real)", 2);
      $this->renderInstrumentRow("NPOL", "NPOL", "Number of polarisations", 2);
      $this->renderInstrumentRow("OSRATIO", "OSRATIO", "Oversampling Ratio", 8);
      $this->renderInstrumentRow("TSAMP", "TSAMP", "Sampling Interval [us]", 8);
      $this->renderInstrumentRow("CHANBW", "CHANBW", "Channel Bandwidth", 8);
      $this->renderInstrumentRow("DSB", "DSB", "Dual Sideband", 8);
      $this->renderInstrumentRow("RESOLUTION", "RESOLUTION", "Byte resolution", 8);
      $this->renderInstrumentRow("INDEPENDENT_BEAMS", "INDEPENDENT_BEAMS", "Beam Independence", 8);
?>
    </table>

    <table class='config' width='100%'>
      <tr>
        <th colspan=6>Stream Configuration</th>
      </tr>
      <tr><th>Stream</th><th>Host</th><th>Beam</th><th>FREQ</th><th>BW</th><th>NCHAN</th></tr>
<?php
      for ($i=0; $i<$this->config["NUM_STREAM"]; $i++)
      {
        list($host, $beam, $subband_id) = explode(":", $this->config["STREAM_".$i]);
        $this->printStreamRow($i, $host, $beam, $subband_id);
      }
?>
    </table>
  </td>

</tr>

<tr> 
  <td colspan=3>
    <input type='button' onClick='javascript:configureButton()' value='Configure'/>
    <input type='button' onClick='javascript:startButton()' value='Start'/>
    <input type='button' onClick='javascript:stopButton()' value='Stop'/>
  </td>
</tr>

</table>
</form>

<iframe name="spip_response" src="" width=80% frameborder=0 height='350px'></iframe>

</center>

<?php
  }

  function renderConfigHeader($title, $id)
  {
    echo "    <table class='config' id='".$id."' border=0 width=100%>\n";
    echo "      <tr><th colspan=2>".$title."</th></tr>\n";
  }

  function renderConfigFooter()
  {
    echo "    </table>\n";
  }

  function renderProcessingModeHeader($title, $id, $show)
  {
    if ($show) 
      $style = "style='display: table;'";
    else
      $style = "style='display: none;'";
    echo "    <table class='config' id='".$id."' ".$style." border=0 width=100%>\n";
    echo "      <tr><th colspan=2>".$title."</th></tr>\n";
  }

  function renderProcessingModeFooter()
  {
    echo "    </table>\n";
  }

  function renderSourceConfig()
  {
    $this->renderConfigHeader("Source Configuration", "source_config");
    $this->renderProcessingRows($this->source_config);
    $this->renderConfigFooter();
  }

  function renderObsConfig()
  {
    $this->renderConfigHeader("Observation Configuration", "obs_config");
    $this->renderProcessingRows($this->obs_config);
    $this->renderConfigFooter();
  }

  function renderBeamConfig()
  {
    $this->renderConfigHeader("Beam Configuration", "beam_config");
    $this->renderProcessingRows($this->beam_config);
    $this->renderConfigFooter();
  }

  function renderCustomConfig()
  {
    $this->renderConfigHeader("Custom Configuration", "custom_config");
    $this->renderProcessingRows($this->custom_config);
    $this->renderConfigFooter();
  }

  function renderProcModes ()
  {
    $this->renderProcessingModeHeader("Processing Modes", "proc_modes", 1);
    $this->renderProcessingRows($this->proc_modes);
    $this->renderProcessingModeFooter();
  }

  function renderFoldMode ()
  {
    $this->renderProcessingModeHeader("Fold Processing Mode Parameters", "fold", 1);
    $this->renderProcessingRows($this->fold_config);
    $this->renderProcessingModeFooter();
  }

  function renderSearchMode ()
  {
    $this->renderProcessingModeHeader("Search Processing Mode Parameters", "search", 0);
    $this->renderProcessingRows($this->search_config);
    $this->renderProcessingModeFooter();
  }

  function renderContinuumMode ()
  {
    $this->renderProcessingModeHeader("Continnum Processing Mode Parameters", "continuum", 0);
    $this->renderProcessingRows($this->continuum_config);
    $this->renderProcessingModeFooter();
  }

  function renderBasebandMode ()
  {
    $this->renderProcessingModeHeader("Baseband Processing Mode Parameters", "baseband", 0);
    $this->renderProcessingRows($this->baseband_config);
    $this->renderProcessingModeFooter();
  }

  function renderSpectralLineMode ()
  {
    $this->renderProcessingModeHeader("Spectral Line Processing Mode Parameters", "spectral_line", 0);
    $this->renderProcessingModeFooter();
  }
  function renderVLBIMode ()
  {
    $this->renderProcessingModeHeader("VLBI Processing Mode Parameters", "vlbi", 0);
    $this->renderProcessingRows($this->vlbi_config);
    $this->renderProcessingModeFooter();
  }

  function boolActive($c, $get)
  {
    $val = "0";
    if (array_key_exists($c["prefix"]."_".$c["tag"], $get))
    {
      if ($get[$c["prefix"]."_".$c["tag"]] == "on")
       $val ="1";
    }
    return $val;
  }

  function generateXMlTag($c, $get)
  {
    if ($c["type"] == "bool")
    {
      $val = $this->boolActive ($c, $get);
    }
    else
    {
      $val = $get[$c["prefix"]."_".$c["tag"]];
    }
    return "<".$c["tag"]." key='".$c["key"]."' ".$c["attr"].">".$val."</".$c["tag"].">";
  }

  function printSPIPResponse($get)
  {
    $xml = "";

    # configuration of beams
    $xml .= "<beam_configuration>";
    foreach ($this->beam_config as $c)
    {
      $xml .= $this->generateXMLTag($c, $get);
    }
    $xml .= "</beam_configuration>";

    if ($get["command"] == "configure")
    {
      $xml .= "<source_parameters>";
      foreach ($this->source_config as $c)
      {
        $xml .= $this->generateXMLTag($c, $get);
      }
      $xml .= "</source_parameters>";

      $xml .= "<observation_parameters>";
      foreach ($this->obs_config as $c)
      {
        $xml .= $this->generateXMLTag($c, $get);
      } 
      $xml .=   "<utc_start key='UTC_START'>None</utc_start>";
      $xml .=   "<utc_stop key='UTC_STOP'>None</utc_stop>";
      $xml .= "</observation_parameters>";

      $xml .= "<custom_parameters>";
      foreach ($this->custom_config as $c)
      {
        $xml .= $this->generateXMLTag($c, $get);
      }
      $xml .= "</custom_parameters>";

      $modes = array();
      $xml .= "<processing_modes>";
      foreach ($this->proc_modes as $c)
      {
        $xml .= $this->generateXMLTag($c, $get);
        if ($this->boolActive($c, $get) == "1")
          array_push ($modes, $c["tag"]);
      }

      $xml .= "</processing_modes>";

      if (in_array("fold", $modes))
      {
        $xml .= "<fold_processing_parameters>";
        foreach ($this->fold_config as $c)
        {
          $xml .= $this->generateXMLTag($c, $get);
        }
        $xml .= "</fold_processing_parameters>";
      }
   
      if (in_array("search", $modes))
      {
        $xml .= "<search_processing_parameters>";
        foreach ($this->search_config as $c)
        {
          $xml .= $this->generateXMLTag($c, $get);
        }
        $xml .= "</search_processing_parameters>";
      }

      if (in_array("continuum", $modes))
      {
        $xml .= "<continuum_processing_parameters>";
        foreach ($this->continuum_config as $c)
        {
          $xml .= $this->generateXMLTag($c, $get);
        }
        $xml .= "</continuum_processing_parameters>";
      }

      if (in_array("spectral_line", $modes))
      {
        $xml .= "<spectral_line_processing_parameters>";
        foreach ($this->spectral_line_config as $c)
        {
          $xml .= $this->generateXMLTag($c, $get);
        }
        $xml .= "</spectral_line_processing_parameters>";
      }

      if (in_array("vlbi", $modes))
      {
        $xml .= "<vlbi_processing_parameters>";
        foreach ($this->vlbi_config as $c)
        {
          $xml .= $this->generateXMLTag($c, $get);
        }
        $xml .= "</vlbi_processing_parameters>";
      }

      if (in_array("baseband", $modes))
      {
        $xml .= "<baseband_processing_parameters>";
        foreach ($this->baseband_config as $c)
        {
          $xml .= $this->generateXMLTag($c, $get);
        }
        $xml .= "</baseband_processing_parameters>";
      }

    }
    else if ($get["command"] == "start")
    {
      $xml .= "<observation_parameters>";
      $xml .=   "<utc_start key='UTC_START'>None</utc_start>";
      $xml .= "</observation_parameters>";
    }
    else if ($get["command"] == "stop")
    {
      $xml .= "<observation_parameters>";
      $xml .=   "<utc_stop key='UTC_STOP'>None</utc_stop>";
      $xml .= "</observation_parameters>";
    }
    else
    {
      echo "ERROR: command [".$get["command"]."] not reconized<br/>\n";
      return;
    }

    // for debug of XML message only
    $testing = 0;
    if ($testing == 1)
    {
      $html = XML_DEFINITION;
      $html .= "<obs_cmd>";
      $html .= $xml;
      $html .= "</obs_cmd>";
    
      header("Content-type: text/xml");
      echo $html;
      return;
    }

    $html = "";

    # If we can have independent control of the beams 
    if ($this->config["INDEPENDENT_BEAMS"] == "true")
    {
      $beam_hosts = array();
      # get the list of hosts for each beam
      for ($i=0; $i<$this->config["NUM_STREAM"]; $i++)
      {
        list($host, $beam, $subband) = explode(":", $this->config["STREAM_".$i]);
        $beam_hosts[$beam] = $host;
      }

      for ($i=0; $i<$this->config["NUM_BEAM"]; $i++)
      {
        if (strcmp($get["beam_state_".$i], "on") !== FALSE)
        {
          $tcs_beam_host = $beam_hosts[$i];
          $tcs_beam_port = $this->config["TCS_INTERFACE_PORT_".$i];
  
          $beam_xml  = "<?xml version='1.0' encoding='ISO-8859-1'?>\n";
          $beam_xml .= "<obs_cmd>\n";
          $beam_xml .= "<command>".$get["command"]."</command>\n";
          
          $beam_xml .= $xml;
          $beam_xml .= "<beam_configuration>\n";
          $beam_xml .=   "<nbeam>1</nbeam>\n";
          
          $beam_xml .=   "<beam_state_0 name='".$this->config["BEAM_".$i]."'>".$get["beam_state_".$i]."</beam_state_0>\n";
          $beam_xml .= "</beam_configuration>\n";

          $beam_xml .= "</obs_cmd>\n";

          $tcs_socket = new spip_socket();
          if ($tcs_socket->open ($tcs_beam_host, $tcs_beam_port) == 0)
          {
            $raw_xml = str_replace("\n","", $beam_xml);
            $tcs_socket->write ($raw_xml."\r\n");
            $reply = $tcs_socket->read();
          }
          else
          {
            $html .= "<p>Could not connect to ".$tcs_beam_host.":".$tcs_beam_port."</p>\n";
          }
          $tcs_socket->close();
        }
      }
    }
    # We have only 1 TCS instance for each beam
    else
    {
      $tcs_host = $this->config["SERVER_HOST"];
      $tcs_port = $this->config["TCS_INTERFACE_PORT"];

      $beam_xml  = "<?xml version='1.0' encoding='ISO-8859-1'?>\n";
      $beam_xml .= "<obs_cmd>\n";
      $beam_xml .= "<command>".$get["command"]."</command>\n";
      $beam_xml .= $xml;
      $beam_xml .= "</obs_cmd>\n";

      $tcs_socket = new spip_socket();
      if ($tcs_socket->open ($tcs_host, $tcs_port) == 0)
      {
        $raw_xml = str_replace("\n","", $beam_xml);
        $tcs_socket->write ($raw_xml."\r\n");
        $reply = $tcs_socket->read();
      }
      else
      {
        $html .= "<p>Could not connect to ".$tcs_host.":".$tcs_port."</p>\n";
      }
      $tcs_socket->close();
    }
    echo $html;
  }


  function renderProcessingRows($array)
  {
    foreach ($array as $c)
    {
      $this->renderProcessingRow($c);
    }
  }

  function renderProcessingRow($c)
  {
    echo "<tr>\n";
    echo "  <td>".$c["name"]."</td>";
    echo "  <td>";
    $id = $c["prefix"]."_".$c["tag"];
    $name = $id;
    if ($c["type"] == "text")
    {
      echo "<input type='text' name='".$name."' id='".$id."' size='".$c["size"]."' value='".$c["value"]."'";
      if ($c["maxlength"] != "")
        echo " maxlength=".$c["maxlength"];
      echo "/>";
    }
    else if ($c["type"] == "bool")
    {
      echo "<input type='checkbox' name='".$name."' id='".$id."'";
      if ($c["value"] == "true")
        echo " checked";
      if ($c["onchange"] != "")
        echo " onchange=\"".$c["onchange"]."\"";
      echo "/>";
    }
    else if ($c["type"] == "radio")
    {
      foreach ($c["value"] as $key => $val)
      {
        $checked = "";
        if ($val == "true") $checked = " checked";
        echo "<input type='radio' name='".$name."' id='".$id."' value='".$key."'".$checked."/>";
        echo "<label for='".$c["tag"]."'>".$key."</label>";
      }
    }
    else if ($c["type"] == "select")
    {
      echo "<select name='".$name."' id='".$id."'>";
      foreach ($c["value"] as $key => $val)
      {
        echo "<option value='".$key."'>".$val."</option>";
      }
      echo "</select>\n";
    }
    else
    {
      echo $c["value"];
    }
    echo "</td>\n";
    echo "</tr>\n";
  }


  function renderInstrumentRow($cfg, $key, $title, $size, $maxlength="")
  {
    $val = "";
    if (array_key_exists($cfg, $this->config))
      $val = $this->config[$cfg];
    $name = strtolower($key);

    echo "<tr>\n";
    echo "  <td>".$title."</td>";
    echo "  <td>".$val."<input type='hidden' name='".$name."' id='".$name."' value='".$val."' size=".$size;
    if ($maxlength != "")
      echo " maxlength=".$maxlength;
    echo "/ readonly></td>\n";
    echo "</tr>\n";
  }

  function printStreamRow($stream, $host, $beam, $subband_id)
  {
    list ($freq, $bw, $nchan) = explode(":", $this->config["SUBBAND_CONFIG_".$subband_id]);
    echo "<tr>\n";
    echo "  <td>".$stream."</td>\n";
    echo "  <td>".$host."</td>\n";
    echo "  <td>".$beam."</td>\n";
    echo "  <td>".$freq."</td>\n";
    echo "  <td>".$bw."</td>\n";
    echo "  <td>".$nchan."</td>\n";
    echo "</tr>\n";
  }
}

if (isset($_GET["command"]))
{
  $obj = new tests();
  $obj->printSPIPResponse($_GET);
}
else
{
  $_GET["single"] = "true";
  handleDirect("tests");
}
