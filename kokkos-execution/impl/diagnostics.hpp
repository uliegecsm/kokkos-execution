#ifndef KOKKOS_EXECUTION_IMPL_DIAGNOSTICS_HPP
#define KOKKOS_EXECUTION_IMPL_DIAGNOSTICS_HPP

/**
 * @file
 *
 * This file contains generic diagnostic utilities.
 */

#include "kokkos-execution/stdexec.hpp"

namespace Kokkos::Execution::Impl {

//! @name Tags to be used to signal an operation state.
template <typename T>
struct WITH_OPSTATE;

//! Tag to be used to signal children errors.
template <typename... T>
struct WITH_CHILDREN_ERRORS;

//! Tag to be used to signal an execution space.
template <typename T>
struct WITH_EXECUTION_SPACE;

//! @name Rebinds the template arguments of @p From onto the template @p To.
///@{
template <template <typename...> class To, template <typename...> class From, typename... As>
auto rebind(From<As...>*) -> To<As...>;

template <template <typename...> class To, typename From>
using rebind_t = decltype(rebind<To>(static_cast<From*>(nullptr)));
///@}

//! @name Diagnosis of @c T if it provides @c diagnose. @note It is considered @c stdexec::__msuccess otherwise.
///@{
template <typename T>
struct TryDiagnose {
    using type = stdexec::__msuccess;
};

template <typename T>
requires std::invocable<decltype(T::diagnose)>
struct TryDiagnose<T> {
    using type = decltype(T::diagnose());
};

template <typename T>
using try_diagnose_t = typename TryDiagnose<T>::type;
///@}

//! @name Diagnose one or more types.
///@{
template <typename...>
struct Diagnose;

template <typename T>
struct Diagnose<T> : TryDiagnose<T> { };

template <typename... Ts>
struct Diagnose {
    using type = stdexec::__mlist<try_diagnose_t<Ts>...>;
};

template <typename... Ts>
requires(sizeof...(Ts) > 0)
using diagnose_t = typename Diagnose<Ts...>::type;
///@}

//! @name Keep the first encountered error, @c stdexec::__msuccess otherwise.
///@{
template <typename...>
struct FirstErrorOf;

template <typename T>
struct FirstErrorOf<T> {
    using type = std::conditional_t<stdexec::__merror<T>, T, stdexec::__msuccess>;
};

template <typename T, typename... Ts>
struct FirstErrorOf<T, Ts...>
    : std::conditional_t<stdexec::__merror<T>, std::type_identity<T>, FirstErrorOf<Ts...>> { };

template <typename T>
struct FirstError : FirstErrorOf<T> { };

template <typename... Ts>
struct FirstError<stdexec::__mlist<Ts...>> : FirstErrorOf<Ts...> { };

template <typename T>
using first_error_t = typename FirstError<T>::type;
///@}

//! @name Collapse a one-item list to the item itself.
///@{
template <typename T>
struct Collapse : std::type_identity<T> { };

template <typename T>
struct Collapse<stdexec::__mlist<T>> {
    using type = T;
};

template <typename T>
using collapse_t = typename Collapse<T>::type;
///@}

/**
 * @name Filter-out anything that is not an error.
 *
 * If no error is found, return @c stdexec::__mlist<> if the input was a list, @c stdexec::__msuccess otherwise.
 * If one error @c E is found, return @c stdexec::__mlist<E> if the input was a list, @c E otherwise.
 * If many errors @c Es... are found, return them as a @c stdexec::__mlist<Es...>.
 */
///@{
template <typename T>
using keep_if_error_t = std::conditional_t<stdexec::__merror<T>, stdexec::__mlist<T>, stdexec::__mlist<>>;

template <typename T>
struct KeepErrors {
    using type = std::conditional_t<stdexec::__merror<T>, T, stdexec::__msuccess>;
};

template <typename... Ts>
struct KeepErrors<stdexec::__mlist<Ts...>> {
    using type = stdexec::__mconcat_result_t<stdexec::__mlist<>, keep_if_error_t<Ts>...>;
};

template <typename List>
using keep_errors_t = typename KeepErrors<List>::type;
///@}

//! If @c T::value is @c false, diagnose.
template <typename T>
constexpr bool mandate_or_explain() noexcept {
    if constexpr (!T::value) {
        static_assert(std::is_same_v<diagnose_t<T>, stdexec::__msuccess>, "Check for the error in the compiler log.");
    }
    return T::value;
}

} // namespace Kokkos::Execution::Impl

#endif // KOKKOS_EXECUTION_IMPL_DIAGNOSTICS_HPP
