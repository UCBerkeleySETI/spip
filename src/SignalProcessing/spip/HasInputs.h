//-*-C++-*-
/***************************************************************************
 *
 *   Copyright (C) 2009 by Willem van Straten
 *   Licensed under the Academic Free License version 2.1
 *
 ***************************************************************************/

#ifndef __spip_HasInputs_h
#define __spip_HasInputs_h

#include <vector>

namespace spip
{
  //! Attaches to Operations with input
  template <class In>
  class HasInputs
  {
  public:

    //! Destructor
    virtual ~HasInputs () {}

    //! Set the container from which input data will be read
    virtual void add_input (const In * _input) 
    {
      inputs.push_back (_input);
    }

    //! Return vector of pointers to the container from which input data will be read
    const std::vector<In *> get_inputs () const { return inputs; }

    //! Returns true if inputs are set
    bool has_inputs() const { return inputs.size() > 0; }

  protected:

    //! Container from which input data will be read
    std::vector<const In *> inputs;
  };
}

#endif

