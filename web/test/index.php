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
    $this->stream_config = $this->get_stream_config();
    $this->custom_config = $this->get_custom_config();
    $this->calibration_config = $this->get_calibration_config();
    $this->proc_modes = $this->get_proc_modes();
    $this->fold_config = $this->get_fold_config();
    $this->search_config = $this->get_search_config();
    $this->continuum_config = $this->get_continuum_config();
    $this->baseband_config = $this->get_baseband_config();
    $this->vlbi_config = $this->get_vlbi_config();
  }
    
  function unique ($prefix, $xml, $key, $name, $type, $value, $size, $attr="", $onchange="")
  {
    return $this->build_hash ($prefix, $xml, $key, $name, $type, "true", $value, $size, $attr, $onchange);
  }

  function custom($prefix, $xml, $key, $name, $type, $value, $size, $attr="", $onchange="")
  {
    return $this->build_hash ($prefix, $xml, $key, $name, $type, "false", $value, $size, $attr, $onchange);
  }

  function build_hash ($prefix, $xml, $key, $name, $type, $unique, $value, $size, $attr="", $onchange="")
  {
    return array ("prefix" => $prefix, "tag" => $xml, "key" => $key, "name" => $name, 
                  "type" => $type, "unique" => $unique, "value" => $value, "attr" => $attr, "size" => $size, 
                  "maxlength" => "", "onchange" => $onchange);
  }

  function get_source_config()
  {
    $a = array();
    array_push($a, $this->unique("source", "name", "SOURCE", "Source Name", "text", "J0437-4715", "16", "epoch='J2000'"));
    array_push($a, $this->unique("source", "ra", "RA", "Right Ascension", "text", "04:37", "16", "units='hh:mm:ss'"));
    array_push($a, $this->unique("source", "dec", "DEC", "Declination", "text", "47:15", "16", "units='dd:mm:ss'"));
    return $a;
  }

  function get_obs_config()
  {
    $a = array();
    array_push($a, $this->unique("obs", "observer", "OBSERVER", "Observer", "text", "Andrew", "16"));
    array_push($a, $this->unique("obs", "project_id", "PID", "Project ID", "text", "P999", "8"));
    array_push($a, $this->unique("obs", "tobs", "TOBS", "Expected Length [s]", "text", "60", "8"));
    array_push($a, $this->unique("obs", "calfreq", "CALFREQ", "Calibrator Frequency [Hz]", "text", "11.123", "8"));
    return $a;
  }

  function get_beam_config()
  {
    $nbeam = $this->config["NUM_BEAM"];
    $a = array();
    array_push($a, $this->unique("beam", "nbeam", "NBEAM", "Number of beams", "text", $nbeam, "4"));
    for ($i=0; $i<$nbeam; $i++)
    {
      array_push($a, $this->unique("beam", "beam_state_".$i, "BEAM_STATE_".$i,  "Beam ".$i, "bool", ($i == 0), "4", "name='".$this->config["BEAM_".$i]."'"));
    }
    return $a;
  }

  function get_stream_config()
  {
    $nstream = $this->config["NUM_STREAM"];
    $a = array();
    array_push($a, $this->unique("stream", "nstream", "NSTREAM", "Number", "hidden", $nstream, "2"));
    array_push($a, $this->custom("stream", "active", "STREAM_ACTIVE", "Active", "bool", "true", "4"));
    return $a;
  }

  function get_custom_config()
  {
    $a = array();
    array_push($a, $this->custom("custom", "adaptive_filter_epsilon", "ADAPTIVE_FILTER_EPSILON", "Adaptive Filter Epsilon", "text", "0.1", "4"));
    array_push($a, $this->custom("custom", "adaptive_filter_nchan", "ADAPTIVE_FILTER_NCHAN", "Adaptive Filter Channels", "text", "128", "5"));
    array_push($a, $this->custom("custom", "adaptive_filter_nsamp", "ADAPTIVE_FILTER_NSAMP", "Adaptive Filter Samples", "text", "1024", "5"));
    return $a;
  }

  function get_calibration_config()
  {
    $a = array();
    array_push($a, $this->unique("calibration", "signal", "CAL_SIGNAL", "Presence", "bool", "false", "4"));
    array_push($a, $this->unique("calibration", "freq", "CAL_FREQ", "Frequnecy", "text", "11", "5", "units='Hertz'"));
    array_push($a, $this->unique("calibration", "phase", "CAL_PHASE", "Starting phase for high state", "text", "0.0", "5"));
    array_push($a, $this->unique("calibration", "duty_cycle", "CAL_DUTY_CYCLE", "Duty cycle for high state", "text", "0.5", "3"));
    array_push($a, $this->unique("calibration", "epoch", "CAL_EPOCH", "Epoch for alignment of CAL", "text", "None", "20", "units='YYYY-DD-MM-HH:MM:SS+0'"));
    array_push($a, $this->unique("calibration", "tsys_avg_time", "TSYS_AVG_TIME", "Averaging time for TSYS estimates", "text", "5", "3", "units='seconds'"));
    array_push($a, $this->unique("calibration", "tsys_freq_resolution", "TSYS_FREQ_RES", "Frequneyc resolution for TSYS estimates [MHz]", "text", "1", "3", "units='MHz'"));
    return $a;
  }

  function get_proc_modes()
  {
    $a = array();
    array_push($a, $this->custom("proc", "fold", "PERFORM_FOLD", "Fold Mode", "bool", "true", "2"));
    array_push($a, $this->custom("proc", "search", "PERFORM_SEARCH", "Search Mode", "bool", "false", "2"));
    array_push($a, $this->custom("proc", "continuum", "PERFORM_CONTINUUM", "Continuum Mode", "bool", "false", "2"));
    array_push($a, $this->custom("proc", "spectral_line", "PERFORM_SPECTRAL_LINE", "Spectral Line Mode", "bool", "false", "2"));
    array_push($a, $this->custom("proc", "vlbi", "PERFORM_VLBI", "VLBI Mode", "bool", "false", "2"));
    array_push($a, $this->custom("proc", "baseband", "PERFORM_BASEBAND", "Baseband Mode", "bool", "false", "2"));
    return $a;
  }

  function get_fold_config()
  {
    $a = array();
    array_push($a, $this->custom("fold", "output_nchan", "FOLD_OUTNCHAN", "Number of output channels", "text", "128", "6"));
    array_push($a, $this->custom("fold", "custom_dm", "FOLD_DM", "Custom DM", "text", "-1", "6"));
    array_push($a, $this->unique("fold", "output_nbin", "FOLD_OUTNBIN", "Number of output phase bins", "text", "1024", "6"));
    array_push($a, $this->unique("fold", "output_tsubint", "FOLD_OUTTSUBINT", "Output subint length [s]", "text", "10", "6"));
    array_push($a, $this->unique("fold", "output_npol", "FOLD_OUTNPOL", "Number of output polarisations", "text", "4", "1"));
    array_push($a, $this->unique("fold", "mode", "MODE", "Observing Type", "radio", array("PSR" => "true", "CAL" => "false"), "6"));
    array_push($a, $this->custom("fold", "sk", "FOLD_SK", "Spectral Kurtosis", "bool", "false", "6"));
    array_push($a, $this->custom("fold", "sk_threshold", "FOLD_SK_THRESHOLD", "Spectral Kurtosis Threshold", "text", "3", "6"));
    array_push($a, $this->custom("fold", "sk_nsamps", "FOLD_SK_NSAMPS", "Spectral Kurtosis Samples", "text", "1024", "6"));
    array_push($a, $this->custom("fold", "append_output", "FOLD_APPEND_OUTPUT", "Append output RF bands", "bool", "false", "6"));
    return $a;
  }

  function get_search_config()
  {
    $a = array();
    array_push($a, $this->custom("search", "output_nchan", "SEARCH_OUTNCHAN", "Output channels", "text", "1024", "6"));
    array_push($a, $this->custom("search", "custom_dm", "SEARCH_DM", "Custom DM", "text", "-1", "6"));
    array_push($a, $this->custom("search", "output_nbit", "SEARCH_OUTNBIT", "Output bits per sample", "text", "8", "2"));
    array_push($a, $this->custom("search", "output_tsamp", "SEARCH_OUTTSAMP", "Output sampling time [us]", "text", "64", "6"));
    array_push($a, $this->custom("search", "output_tsubint", "SEARCH_OUTTSUBINT", "Output subint length [s]", "text", "10", "6"));
    array_push($a, $this->custom("search", "output_npol", "SEARCH_OUTNPOL", "Number of output polarisations", "text", "1", "1"));
    array_push($a, $this->custom("search", "coherent_dedispersion", "SEARCH_COHERENT_DEDISPERSION", "Perform coherent dedispersion", "bool", "false", "6"));
    return $a;
  }

  function get_continuum_config()
  {
    $a = array();
    array_push($a, $this->custom("continuum", "output_nchan", "CONTINUUM_OUTNCHAN", "Output channels", "text", "32768", "6"));
    array_push($a, $this->custom("continuum", "output_tsamp", "CONTINUUM_OUTTSAMP", "Output sampling time [s]", "text", "1", "6"));
    array_push($a, $this->custom("continuum", "output_tsubint", "CONTINUUM_OUTTSUBINT", "Output subint length [s]", "text", "10", "6"));
    array_push($a, $this->custom("continuum", "output_npol", "CONTINUUM_OUTNPOL", "Number of output polarisations", "text", "4", "1"));
    return $a;
  }

  function get_vlbi_config()
  {
    $a = array();
    array_push($a, $this->unique("vlbi", "auto_gain", "VLBI_AUTO_GAIN", "Automatic gain control", "bool", "true", "8"));
    array_push($a, $this->unique("vlbi", "level_setting", "VLBI_LEVEL_SETTING", "Level setting options", "select", array("1" => "No Level Setting", "1" => "Adaptive", "2" => "Constant"), "8"));
    array_push($a, $this->unique("vlbi", "level_time_scale", "VLBI_TIME_SCALE", "Level setting timescale", "text", 1, "8"));
    array_push($a, $this->unique("vlbi", "output_nbit", "OUTNBIT", "Output bits per sample", "text", "8", "2"));
    array_push($a, $this->unique("vlbi", "output_bw", "OUTBW", "Output bandwidth", "select", array("16" => "16 MHz", "32" => "32 MHz", "64" => "64 MHz", "128" => "128 MHz"), "8"));
    return $a;
  }


  function get_baseband_config()
  {
    $a = array();
    array_push($a, $this->unique("baseband", "output_nbit", "OUTNBIT", "Output bits per sample", "text", "16", "2"));
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

      function enableSubbands(area, id)
      {
        var re_str = "^" + area + "_[a-z_]*_" + id + "$"
        var re = new RegExp(re_str)
        var elms = document.getElementsByClassName("form");
        for (i=0; i<elms.length; i++)
        {
          if (re.test(elms[i].id))
          {
            document.getElementById(elms[i].id).style.display = "inline";
          }
        }
      }

      function updateStreams (id)
      {
        var ele = document.getElementById(id)
        var ele_tag_name = ele.tagName.toLowerCase();
        var re_str = "^" + id + "_[0-9]*$";
        var re = new RegExp(re_str);
        var elms = document.getElementsByClassName("form");
        for (i=0; i<elms.length; i++)
        {
          if (re.test(elms[i].id))
          {
            if (ele_tag_name == "input")
            {
              if (ele.type == "text")
              {
                document.getElementById(elms[i].id).value = ele.value;
              }
              else if (ele.type == "checkbox")
              {
                document.getElementById(elms[i].id).checked = ele.checked;
              }
              else if (ele.type == "radio")
              {
                document.getElementById(elms[i].id).selectedIndex = ele.selectedIndex;
              }
              else  
              {
              }
            }
            else if (ele_tag_name == "select")
            {
              document.getElementById(elms[i].id).selectedIndex = ele.selectedIndex;
            }
            else  
            {
            }
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

<form name="tcs" target="spip_response" action="" method="post">

<input type="hidden" name="command" id="command" value="">
<table cellpadding='3px' border=0 cellspacing=20px width='100%'>

<tr>
<?php
  echo "<td valign=top>\n";
  $this->renderSourceConfig();
  echo "</td>\n";
  echo "<td valign=top>\n";
  $this->renderObsConfig();
  echo "</td>\n";
  echo "</tr>\n";
  echo "<tr>\n";
  echo "<td valign=top>\n";
  $this->renderBeamConfig();
  echo "</td>\n";
  echo "<td valign=top>\n";
  $this->renderCalConfig();
  echo "</td>\n";
?>
</tr>
<tr>
  <td colspan=2>
<?php
  $this->renderStreamConfig();
  $this->renderCustomConfig();
  $this->renderProcModes();
  $this->renderFoldMode();
  $this->renderSearchMode();
  $this->renderContinuumMode();
  $this->renderSpectralLineMode();
  $this->renderVLBIMode();
  $this->renderBasebandMode();
?>
  </td>
</tr>
<tr>
  <td valign=top>
<?php  
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
  </td>
  <td colspan=2 valign=top>

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
  <td>
    <input type='button' onClick='javascript:configureButton()' value='Configure'>
    <input type='button' onClick='javascript:startButton()' value='Start'>
    <input type='button' onClick='javascript:stopButton()' value='Stop'>
  </td>
</tr>

</table>
</form>

<iframe name="spip_response" src="" width='80%' frameborder=0 height='350px'></iframe>

</center>

<?php
  }

  function renderConfigHeader($title, $id)
  {
    echo "    <table class='config' id='".$id."' border='0' width='100%'>\n";
    echo "      <tr><th colspan='2'>".$title."</th></tr>\n";
  }

  function renderConfigFooter()
  {
    echo "    </table>\n";
  }

  function renderProcessingModeHeader($title, $id)
  {
    echo "    <table class='config' id='".$id."' border='0' width='100%'>\n";
    if ($this->config["NUM_STREAM"] > 1)
    {
      echo "      <tr><th colspan=".($this->config["NUM_STREAM"] + 2).">".$title."</th></tr>\n";
      echo "      <tr><th align='left' width='200px'>Parameter</th><th align='left' width='100px'>Default</th>";
      for ($i=0; $i<$this->config["NUM_STREAM"]; $i++)
        echo "<th align=left><a href='javascript:enableSubbands(\"".$id."\",".$i.")' font-color='black'>".$i."</a></th>";
      echo "</tr>\n";
    }
    else
    {
      echo "      <tr><th colspan=3>".$title."</th></tr>\n";
      echo "      <tr><th align='left' width='200px'>Parameter</th><th align='left' width='100px'>Default</th>";
    }
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

  function renderStreamConfig()
  {
    $this->renderProcessingModeHeader("Stream Configuration", "stream");
    $this->renderProcessingRows($this->stream_config);
    $this->renderConfigFooter();
  }

  function renderCustomConfig()
  {
    $this->renderProcessingModeHeader("Custom Configuration", "custom");
    $this->renderProcessingRows($this->custom_config);
    $this->renderConfigFooter();
  }

  function renderCalConfig()
  {
    $this->renderConfigHeader("Calibration Configuration", "calibration");
    $this->renderProcessingRows($this->calibration_config);
    $this->renderConfigFooter();
  }

  function renderProcModes ()
  {
    $this->renderProcessingModeHeader("Processing Modes", "proc");
    $this->renderProcessingRows($this->proc_modes);
    $this->renderProcessingModeFooter();
  }

  function renderFoldMode ()
  {
    $this->renderProcessingModeHeader("Fold Processing Mode Parameters", "fold");
    $this->renderProcessingRows($this->fold_config);
    $this->renderProcessingModeFooter();
  }

  function renderSearchMode ()
  {
    $this->renderProcessingModeHeader("Search Processing Mode Parameters", "search");
    $this->renderProcessingRows($this->search_config);
    $this->renderProcessingModeFooter();
  }

  function renderContinuumMode ()
  {
    $this->renderProcessingModeHeader("Continuum Processing Mode Parameters", "continuum");
    $this->renderProcessingRows($this->continuum_config);
    $this->renderProcessingModeFooter();
  }

  function renderBasebandMode ()
  {
    $this->renderProcessingModeHeader("Baseband Processing Mode Parameters", "baseband");
    $this->renderProcessingRows($this->baseband_config);
    $this->renderProcessingModeFooter();
  }

  function renderSpectralLineMode ()
  {
    $this->renderConfigHeader("Spectral Line Processing Mode Parameters", "spectral_line");
    $this->renderProcessingModeFooter();
  }
  function renderVLBIMode ()
  {
    $this->renderProcessingModeHeader("VLBI Processing Mode Parameters", "vlbi");
    $this->renderProcessingRows($this->vlbi_config);
    $this->renderProcessingModeFooter();
  }

  function boolActive($c, $get, $suffix)
  {
    $val = "0";
    if (array_key_exists($c["prefix"]."_".$c["tag"].$suffix, $get))
    {
      if ($get[$c["prefix"]."_".$c["tag"].$suffix] == "on")
       $val ="1";
    }
    return $val;
  }

  function generateXMlTag($c, $get, $stream)
  {
    if ($c["unique"] == "true")
    {
      if ($c["type"] == "bool")
      {
        $val = $this->boolActive ($c, $get, "");
      }
      else
      {
        $val = $get[$c["prefix"]."_".$c["tag"]];
      }
      #return "<".$c["tag"]." key='".$c["key"]."' ".$c["attr"].">".$val."</".$c["tag"].">";
      $xml = "<".$c["tag"]." key='".$c["key"]."'";
      if ($c["attr"] != "")
        $xml .= " ".$c["attr"];
      $xml .= ">".$val."</".$c["tag"].">";
      return $xml;
    }
    else
    {
      if ($c["type"] == "bool")
      {
        $val = $this->boolActive ($c, $get, "_".$stream);
      }
      else
      {
        $val = $get[$c["prefix"]."_".$c["tag"]."_".$stream];
      }
      #return "<".$c["tag"]." key='".$c["key"]."' ".$c["attr"].">".$val."</".$c["tag"].">";
      $xml = "<".$c["tag"]." key='".$c["key"]."'";
      if ($c["attr"] != "")
        $xml .= " ".$c["attr"];
      $xml .= ">".$val."</".$c["tag"].">";
      return $xml;
    }
  }

  function printSPIPResponse($get)
  {
    //print_r($get);

    $xml = "";

    # configuration of beams
    $xml .= "<beam_configuration>";
    foreach ($this->beam_config as $c)
    {
      $xml .= $this->generateXMLTag($c, $get, "");
    }
    $xml .= "</beam_configuration>";

    # configuration of streams
    $xml .= "<stream_configuration>";
    $xml .= $this->generateXMLTag($this->stream_config[0], $get, "");
    for ($i=0; $i<$this->config["NUM_STREAM"]; $i++)
    {
      $xml .= $this->generateXMLTag($this->stream_config[1], $get, $i);
    }
    $xml .= "</stream_configuration>";

    if ($get["command"] == "configure")
    {
      $xml .= "<source_parameters>";
      foreach ($this->source_config as $c)
      {
        $xml .= $this->generateXMLTag($c, $get, "");
      }
      $xml .= "</source_parameters>";

      $xml .= "<observation_parameters>";
      foreach ($this->obs_config as $c)
      {
        $xml .= $this->generateXMLTag($c, $get, "");
      }
      $xml .=   "<utc_start key='UTC_START'>None</utc_start>";
      $xml .=   "<utc_stop key='UTC_STOP'>None</utc_stop>";
      $xml .= "</observation_parameters>";

      $xml .= "<calibration_parameters>";
      foreach ($this->calibration_config as $c)
      {
        $xml .= $this->generateXMLTag($c, $get, "");
      }
      $xml .= "</calibration_parameters>";

      # repeat these for all sub-bands
      for ($i=0; $i<$this->config["NUM_STREAM"]; $i++)
      {
        $xml .= "<stream".$i.">";

        $xml .= "<custom_parameters>";
        foreach ($this->custom_config as $c)
        {
          $xml .= $this->generateXMLTag($c, $get, $i);
        }
        $xml .= "</custom_parameters>";

        $modes = array();
        $xml .= "<processing_modes>";
        foreach ($this->proc_modes as $c)
        {
          $xml .= $this->generateXMLTag($c, $get, $i);
          for ($j=0; $j<$this->config["NUM_STREAM"]; $j++)
            if ($this->boolActive($c, $get, "_".$j) == "1")
              array_push ($modes, $c["tag"]);
        }
        $xml .= "</processing_modes>";

        if (in_array("fold", $modes))
        {
          $xml .= "<fold_processing_parameters>";
          foreach ($this->fold_config as $c)
          {
            $xml .= $this->generateXMLTag($c, $get, $i);
          }
          $xml .= "</fold_processing_parameters>";
        }
     
        if (in_array("search", $modes))
        {
          $xml .= "<search_processing_parameters>";
          foreach ($this->search_config as $c)
          {
            $xml .= $this->generateXMLTag($c, $get, $i);
          }
          $xml .= "</search_processing_parameters>";
        }

        if (in_array("continuum", $modes))
        {
          $xml .= "<continuum_processing_parameters>";
          foreach ($this->continuum_config as $c)
          {
            $xml .= $this->generateXMLTag($c, $get, $i);
          }
          $xml .= "</continuum_processing_parameters>";
        }

        if (in_array("spectral_line", $modes))
        {
          $xml .= "<spectral_line_processing_parameters>";
          foreach ($this->spectral_line_config as $c)
          {
            $xml .= $this->generateXMLTag($c, $get, $i);
          }
          $xml .= "</spectral_line_processing_parameters>";
        }

        if (in_array("vlbi", $modes))
        {
          $xml .= "<vlbi_processing_parameters>";
          foreach ($this->vlbi_config as $c)
          {
            $xml .= $this->generateXMLTag($c, $get, $i);
          }
          $xml .= "</vlbi_processing_parameters>";
        }

        if (in_array("baseband", $modes))
        {
          $xml .= "<baseband_processing_parameters>";
            foreach ($this->baseband_config as $c)
          {
            $xml .= $this->generateXMLTag($c, $get, $i);
          }
          $xml .= "</baseband_processing_parameters>";
        }
        $xml .= "</stream".$i.">";
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
    
      header("content-type: text/xml");
      echo $html;
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
    if ($c["unique"] == "false")
      $this->renderProcessingCell ($id, $name, $c, "onchange='updateStreams(\"".$id."\")'");
    else
      $this->renderProcessingCell ($id, $name, $c, "");
    echo "</td>\n";

    for ($i=0; $i<$this->config["NUM_STREAM"]; $i++)
    {
      echo "  <td>";
      if ($c["unique"] == "false")
      {
        $id = $c["prefix"]."_".$c["tag"]."_".$i;
        $name = $id;
        $this->renderProcessingCell ($id, $name, $c, "style='display: none;'");
      }
      else
      {
        echo "<span id='".$id."_".$i."' style='display: none;'>N/A</span>";
      }
      echo "</td>\n";
    }
    
    echo "</tr>\n";
  }

  function renderProcessingCell($id, $name, $c, $s)
  {
    if ($c["type"] == "text")
    {
      echo "<input class='form' type='text' ".$s." name='".$name."' id='".$id."' size='".$c["size"]."' value='".$c["value"]."'";
      if ($c["maxlength"] != "")
        echo " maxlength=".$c["maxlength"];
      echo ">";
    }
    else if ($c["type"] == "bool")
    {
      echo "<input class='form' type='checkbox' ".$s." name='".$name."' id='".$id."'";
      if ($c["value"] == "true")
        echo " checked";
      if ($c["onchange"] != "")
        echo " onchange=\"".$c["onchange"]."\"";
      echo ">";
    }
    else if ($c["type"] == "radio")
    {
      $idx = 0;
      foreach ($c["value"] as $key => $val)
      {
        $checked = "";
        if ($val == "true") $checked = " checked";
        echo "<input class='form' type='radio' ".$s." name='".$name."' id='".$id.$idx."' value='".$key."'".$checked.">";
        echo "<label class='form' for='".$id.$idx."'>".$key."</label>";
        $idx++;
      }
    }
    else if ($c["type"] == "select")
    {
      echo "<select class='form' name='".$name."' id='".$id."' ".$s.">";
      foreach ($c["value"] as $key => $val)
      {
        echo "<option value='".$key."'>".$val."</option>";
      }
      echo "</select>\n";
    }
    else if ($c["type"] == "hidden")
    {
      echo "<input class='form' type='hidden' name='".$name."' id='".$id."' value='".$c["value"]."'>".$c["value"];
    }
    else
    {
      echo $c["value"];
    }
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
    echo " readonly></td>\n";
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

if (isset($_POST["command"]))
{
  $obj = new tests();
  $obj->printSPIPResponse($_POST);
}
else
{
  $_GET["single"] = "true";
  handleDirect("tests");
}
