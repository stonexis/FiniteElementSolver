#pragma once
#include <cstddef>

namespace math_helper {
    template<typename ScalarT>
    [[nodiscard]] constexpr ScalarT fast_pow(ScalarT base, std::size_t exp) noexcept {
        if (exp == 0)
            return ScalarT(1);

        ScalarT result = ScalarT(1);
        while (exp > 0) {
            if (exp & 1)
                result *= base;

            base *= base;
            exp >>= 1;
        }

        return result;
    }
}