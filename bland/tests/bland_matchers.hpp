#pragma once

#include <catch2/matchers/catch_matchers_templated.hpp>

template<typename T>
struct WithinAbsMatcher : Catch::Matchers::MatcherGenericBase {
    WithinAbsMatcher(std::vector<T> const& range, T tol):
        range{ range }, tol(tol)
    {}

    template<typename O>
    bool match(std::vector<O> const& other) const {
        using std::begin; using std::end;

        return std::equal(begin(range), end(range), begin(other), end(other),
        [this](const T &a, const O &b) -> bool {
            return std::abs(a-b) < this->tol;
        });
    }

    std::string describe() const override {
        return "Equals: " + Catch::rangeToString(range);
    }

private:
    std::vector<T> const& range;
    T tol;
};

template<typename T>
auto BlandWithinAbs(const std::vector<T>& range, T tol) -> WithinAbsMatcher<T> {
    return WithinAbsMatcher<T>{range, tol};
}