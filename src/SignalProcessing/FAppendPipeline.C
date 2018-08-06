/***************************************************************************
 *
 *   Copyright (C) 2018 Andrew Jameson
 *   Licensed under the Academic Free License version 2.1
 *
 ***************************************************************************/

#include "spip/FAppendPipeline.h"

#include <signal.h>
#include <unistd.h>
#include <cstring>
#include <iostream>
#include <stdexcept>
#include <new>

//#define _DEBUG

using namespace std;

spip::FAppendPipeline::FAppendPipeline (vector<string> in_keys, string out_key)
{
  output_state = spip::Signal::Coherence;

  n_inputs = in_keys.size();
  in_dbs.resize(n_inputs);
  for (unsigned i=0; i<n_inputs; i++)
  {
    in_dbs[i] = new DataBlockRead(in_keys[i].c_str());
  }
  out_db = new DataBlockWrite (out_key.c_str());

  for (unsigned i=0; i<n_inputs; i++)
  {
    in_dbs[i]->connect();
    in_dbs[i]->lock();
  }
  out_db->connect();
  out_db->lock();

  verbose = false;
}

spip::FAppendPipeline::~FAppendPipeline()
{
  for (unsigned i=0; i<n_inputs; i++)
  { 
    in_dbs[i]->unlock();
    in_dbs[i]->disconnect();
    delete in_dbs[i];
  }

  out_db->unlock();
  out_db->disconnect();
  delete out_db;
}

//! build the pipeline containers and transforms
void spip::FAppendPipeline::configure ()
{
  if (verbose)
    cerr << "spip::FAppendPipeline::configure ()" << endl;

  inputs.resize (n_inputs);
  for (unsigned i=0; i<n_inputs; i++)
  {
    inputs[i] = new spip::ContainerRingRead (in_dbs[i]);
  }

  output = new spip::ContainerRingWrite (out_db);

  appender = new spip::AppendFrequencyRAM ();
  for (unsigned i=0; i<n_inputs; i++)
  {
    appender->add_input(inputs[i]);
  }
  appender->set_output (output);
  appender->set_verbose (verbose); 
}

//! process meta-data through the pipeline, performing all resource allocation
void spip::FAppendPipeline::open ()
{
  if (verbose)
    cerr << "spip::FAppendPipeline::open()" << endl;

  // read from the inputs
  for (unsigned i=0; i<n_inputs; i++)
  {
    if (verbose)
      cerr << "spip::FAppendPipeline::open inputs[" << i << "]->read_header()" << endl;
    inputs[i]->process_header();
  }

  // configure the appender
  if (verbose)
    cerr << "spip::FAppendPipeline::open appender->configure(TFPS)" << endl;
  appender->configure (spip::Ordering::TFPS);

  // write the output header
  if (verbose)
    cerr << "spip::FAppendPipeline::open output->process_header()" << endl;
  output->process_header();
}

//! close the input and output data blocks
void spip::FAppendPipeline::close ()
{
  if (verbose)
    cerr << "spip::FAppendPipeline::close()" << endl;

  // todo, remove reference to DBs
  for (unsigned i=0; i<n_inputs; i++)
  {
    // close the input buffer 
    if (in_dbs[i]->is_block_open())
    {
      if (verbose)
        cerr << "spip::FAppendPipeline::close in_dbs[" << i << "]->close_block("
             << in_dbs[i]->get_data_bufsz() << ")" << endl;
      in_dbs[i]->close_block (in_dbs[i]->get_data_bufsz());
    }

    // close the input data block
    if (verbose)
      cerr << "spip::FAppendPipeline::close in_dbs[" << i << "]->close()" << endl;
    in_dbs[i]->close();
  }

}

// process blocks of input data until the end of the data stream
bool spip::FAppendPipeline::process ()
{
  if (verbose)
    cerr << "spip::FAppendPipeline::process ()" << endl;

  bool keep_processing = true;

  vector<uint64_t> input_bufszs(n_inputs);
  vector<uint64_t> nbytes_inputs(n_inputs);
  for (unsigned i=0; i<n_inputs; i++)
  {
    input_bufszs[i] = in_dbs[i]->get_data_bufsz();
    nbytes_inputs[i] = 0;
  } 

  while (keep_processing)
  {
    // read a blocks of input data
    for (unsigned i=0; i<n_inputs; i++)
    {
      if (verbose)
        cerr << "spip::FAppendPipeline::process inputs[" << i << "]->open_block()" << endl;
      nbytes_inputs[i] = inputs[i]->open_block();

      if (nbytes_inputs[i] < input_bufszs[i])
        keep_processing = false;
    }

    // only process full blocks of data
    if (keep_processing)
    {
      if (verbose)
        cerr << "spip::FAppendPipeline::process output->open_block()" << endl;
      output->open_block();

      if (verbose)
        cerr << "spip::FAppendPipeline::process appender->prepare()" << endl;
      appender->prepare ();
      if (verbose)
        cerr << "spip::FAppendPipeline::process appender->combination()" << endl;
      appender->combination();

      if (verbose)
        cerr << "spip::FAppendPipeline::process output->close_block()" << endl;
      output->close_block();
    }

    for (unsigned i=0; i<n_inputs; i++)
    {
      if (verbose)
        cerr << "spip::FAppendPipeline::process inputs[" << i << "]->close_block()" << endl;
      inputs[i]->close_block();
    }
  }

  // close the data blocks
  close();

  return true;
}
