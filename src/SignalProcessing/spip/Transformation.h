/***************************************************************************
 *
 *   Copyright (C) 2016 Andrew Jameson
 *   Licensed under the Academic Free License version 2.1
 *
 ***************************************************************************/

#ifndef __Transformation_h
#define __Transformation_h

#include "spip/HasInput.h"
#include "spip/HasOutput.h"
#include "spip/Error.h"

#include <iostream> 

namespace spip {

  //! All Transformation must define their behaviour
  typedef enum { inplace, outofplace } Behaviour;

  template <class In, class Out>
  class Transformation : public HasInput<In>, public HasOutput<Out>
  {
    public:

      Transformation (const char * _name, Behaviour _type);

      virtual ~Transformation ();

      void set_input (const In* input);

      void set_output (Out * output);

      void set_verbose (bool _verbose) { verbose = _verbose; };

      //! Return the unique name of this operation
      std::string get_name() const { return operation_name; }

      std::string name (const std::string& function)
      { return "spip::Transformation["+get_name()+"]::" + function; }

    protected:

      //! Return false if the input doesn't have enough data to proceed
      virtual bool can_operate();

      //! Define the Operation pure virtual method
      //virtual void operation ();

      //! Declare that sub-classes must define a transformation method
      virtual void transformation () = 0;

      //! Configures transformation once, when input is known
      virtual void configure (Ordering output_order) = 0;

      //! Prepares each call to transformation, called just prior
      virtual void prepare () = 0;

      //! Ensure meta-data is correct in output
      virtual void prepare_output () = 0;

      //! Return the type of transformation
      Behaviour get_type () { return type; };

      bool verbose;

    private:

      //! Unique name of this operation
      std::string operation_name;

      //! Behaviour of Transformation
      Behaviour type;

  };
}

//! All sub-classes must specify name and capacity for inplace operation
template<class In, class Out>
spip::Transformation<In,Out>::Transformation (const char* _name, Behaviour _type)
{
  verbose = false;
  type = _type;
}

//! Return false if the input doesn't have enough data to proceed
template<class In, class Out>
bool spip::Transformation<In,Out>::can_operate()
{
  if (!this->has_input())
    return false;

  return true;
}

template <class In, class Out>
void spip::Transformation<In, Out>::set_input (const In* _input)
{
  if (verbose)
    std::cerr << "spip::Transformation["+this->get_name()+"]::set_input ("<<_input<<")"<<std::endl;

  this->input = _input;

  if ( type == outofplace && this->input && this->output
       && (const void*)this->input == (const void*)this->output )
    throw Error (InvalidState, "spip::Transformation["+this->get_name()+"]::set_input",
     "input must != output");

  if( type==inplace )
    this->output = (Out*)_input;
}

template <class In, class Out>
void spip::Transformation<In, Out>::set_output (Out* _output)
{
  if (verbose)
    std::cerr << "spip::Transformation["+this->get_name()+"]::set_output ("<<_output<<")"<<std::endl;

  if (type == inplace && this->input
      && (const void*)this->input != (const void*)_output )
    throw Error (InvalidState, "spip::Transformation["+this->get_name()+"]::set_output",
     "inplace transformation input must equal output");

  if ( type == outofplace && this->input && this->output
       && (const void*)this->input == (const void*)_output )
    throw Error (InvalidState, "spip::Transformation["+this->get_name()+"]::set_output",
     "output must != input");

  this->output = _output;

  if( type == inplace && !this->has_input() )
    this->input = (In*)_output;

}

template <class In, class Out>
spip::Transformation<In,Out>::~Transformation()
{
  if (verbose)
    std::cerr << name("dtor") << std::endl;
}

     
#endif 
