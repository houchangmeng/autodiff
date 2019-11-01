//                  _  _
//  _   _|_ _  _|o_|__|_
// (_||_||_(_)(_|| |  |
//
// automatic differentiation made easier in C++
// https://github.com/autodiff/autodiff
//
// Licensed under the MIT License <http://opensource.org/licenses/MIT>.
//
// Copyright (c) 2018-2019 Allan Leal
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in all
// copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.

#pragma once

// C++ includes
#include <type_traits>
#include <vector>

namespace autodiff {
namespace detail {

//-------------------------------------------------------------------------------------------------
// Basic Traits (traits/basic.hpp)
//-------------------------------------------------------------------------------------------------
template<bool value>
using EnableIf = typename std::enable_if<value>::type;

template<typename T>
using PlainType = typename std::remove_cv<typename std::remove_reference<T>::type>::type;

template<typename A, typename B>
using CommonType = typename std::common_type<A, B>::type;


//-------------------------------------------------------------------------------------------------
// Number Traits (traits/number.hpp)
//-------------------------------------------------------------------------------------------------

template<typename T>
constexpr bool isNumber = std::is_arithmetic<PlainType<T>>::value;

template<typename T>
struct NumericTypeInfoNotDefinedFor { using type = T; };

template<typename T>
struct NumberTraits
{
    /// The underlying floating point type of the autodiff number type.
    using NumericType = std::conditional_t<isNumber<T>, T, NumericTypeInfoNotDefinedFor<T>>;

    /// The order of the autodiff number type.
    static constexpr auto Order = 0;
};

template<typename T>
using NumericType = typename NumberTraits<PlainType<T>>::NumericType;

template<typename T>
constexpr auto Order = NumberTraits<PlainType<T>>::Order;



template<typename V>
struct VectorTraitsNotDefinedFor {};

template<typename V>
struct VectorTraits
{
    using ValueType = VectorTraitsNotDefinedFor<V>;
    using ReplaceValueType = VectorTraitsNotDefinedFor<V>;
};

template<typename T, template<class> typename Allocator>
struct VectorTraits<std::vector<T, Allocator<T>>>
{
    using ValueType = T;

    template<typename NewValueType>
    using ReplaceValueType = std::vector<NewValueType, Allocator<NewValueType>>;
};

template<typename V>
using VectorValueType = typename VectorTraits<PlainType<V>>::ValueType;

template<typename V, typename NewValueType>
using VectorReplaceValueType = typename VectorTraits<V>::template ReplaceValueType<NewValueType>;


//-------------------------------------------------------------------------------------------------
// Derivative Type Support Traits (traits/derivativesupport.hpp)
//-------------------------------------------------------------------------------------------------

//-------------------------------------------------------------------------------------------------
// Class Member Traits (traits/classmembers.hpp)
//-------------------------------------------------------------------------------------------------

//==========================================================================================================================================================
// The code below was taken from: https://stackoverflow.com/questions/87372/check-if-a-class-has-a-member-function-of-a-given-signature/16867422#16867422
// It implements checkers to determine if a type has a member variable, function, etc.
//==========================================================================================================================================================

template <typename... Args> struct ambiguate : public Args... {};

template<typename A, typename = void>
struct got_type : std::false_type {};

template<typename A>
struct got_type<A> : std::true_type { typedef A type; };

template<typename T, T>
struct sig_check : std::true_type {};

template<typename Alias, typename AmbiguitySeed>
struct has_member {
    template<typename C> static char ((&f(decltype(&C::value))))[1];
    template<typename C> static char ((&f(...)))[2];

    //Make sure the member name is consistently spelled the same.
    static_assert(
        (sizeof(f<AmbiguitySeed>(0)) == 1)
        , "Member name specified in AmbiguitySeed is different from member name specified in Alias, or wrong Alias/AmbiguitySeed has been specified."
    );

    static bool const value = sizeof(f<Alias>(0)) == 2;
};

//Check for any member with given name, whether var, func, class, union, enum.
#define CREATE_MEMBER_CHECK(member)                                         \
                                                                            \
template<typename T, typename = std::true_type>                             \
struct Alias_##member;                                                      \
                                                                            \
template<typename T>                                                        \
struct Alias_##member <                                                     \
    T, std::integral_constant<bool, got_type<decltype(&T::member)>::value>  \
> { static const decltype(&T::member) value; };                             \
                                                                            \
struct AmbiguitySeed_##member { char member; };                             \
                                                                            \
template<typename T>                                                        \
struct has_member_##member {                                                \
    static const bool value                                                 \
        = has_member<                                                       \
            Alias_##member<ambiguate<T, AmbiguitySeed_##member>>            \
            , Alias_##member<AmbiguitySeed_##member>                        \
        >::value                                                            \
    ;                                                                       \
}

// Create type trait struct `has_member_size`.
CREATE_MEMBER_CHECK(size);

/// Boolean constant that is true if type T implements `size` method.
template<typename T>
constexpr bool hasSize = has_member_size<PlainType<T>>::value;


} // namespace detail

using detail::NumericType;

} // namespace autodiff
