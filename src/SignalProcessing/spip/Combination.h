/***************************************************************************
 *
 *   Copyright (C) 2018 Andrew Jameson
 *   Licensed under the Academic Free License version 2.1
 *
 ***************************************************************************/

#ifndef __Combination_h
#define __Combination_h

#include "spip/HasInputs.h"
#include "spip/HasOutput.h"
#include "spip/Error.h"

#include <iostream> 
#include <vector> 

namespace spip {

  template <class In, class Out>
  class Combination : public HasInputs<In>, public HasOutput<Out>
  {
    public:

      Combination (const char * _name);

      virtual ~Combination ();

      void add_input (const In* input);

      void set_output (Out * output);

      void set_verbose (bool _verbose) { verbose = _verbose; };

      //! Return the unique name of this operation
      std::string get_name() const { return operation_name; }

      std::string name (const std::string& function)
      { return "spip::Combination["+get_name()+"]::" + function; }

    protected:

      //! Return false if the input doesn't have enough data to proceed
      virtual bool can_operate();

      //! Define the Operation pure virtual method
      //virtual void operation ();

      //! Declare that sub-classes must define a combination method
      virtual void combination () = 0;

      //! Configures combination once, when input is known
      virtual void configure (Ordering output_order) = 0;

      //! Prepares each call to combination, called just prior
      virtual void prepare () = 0;

      //! Ensure meta-data is correct in output
      virtual void prepare_output () = 0;

      bool verbose;

    private:

      //! Unique name of this operation
      std::string operation_name;

  };
}

//! All sub-classes must specify name and capacity for inplace operation
template<class In, class Out>
spip::Combination<In, Out>::Combination (const char* _name)
{
  verbose = false;
}

//! Return false if the input doesn't have enough data to proceed
template<class In, class Out>
bool spip::Combination<In,Out>::can_operate()
{
  if (!this->has_inputs())
    return false;

  return true;
}

template <class In, class Out>
void spip::Combination<In, Out>::add_input (const In* _input)
{
  if (verbose)
    std::cerr << "spip::Combination["+this->get_name()+"]::add_input (" << _input << ")" << std::endl;
  this->inputs.push_back(_input);
}

template <class In, class Out>
void spip::Combination<In, Out>::set_output (Out* _output)
{
  if (verbose)
    std::cerr << "spip::Combination["+this->get_name()+"]::set_output ("<<_output<<")"<<std::endl;

  this->output = _output;
}

template <class In, class Out>
spip::Combination<In, Out>::~Combination()
{
  if (verbose)
    std::cerr << name("dtor") << std::endl;
}

     
#endif 
