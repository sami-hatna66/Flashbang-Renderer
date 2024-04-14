#include <lightwave.hpp>

#include <functional>
#include "pcg32.h"
#include <random>

namespace lightwave {

class Halton : public Sampler {
private:
    int m_prime_index;
    int m_sample_index;
    std::vector<int> m_rad_inv_perms;
    const Point2i m_base_scales = {2, 3};
    const Point2i m_base_exps = {1, 1};
    float m_mask = 0;

    pcg32 m_pcg;

    void populateRadInvPerms() {
        int perm_vec_size = 0;
        for (auto prime : Primes) perm_vec_size += prime;
        m_rad_inv_perms.resize(perm_vec_size);
        int *p = &m_rad_inv_perms[0];
        for (size_t i = 0; i < Primes.size(); ++i) {
            for (int j = 0; j < Primes[i]; ++j) p[j] = j;
            p += Primes[i];
        }
    }

    int multiplicativeInverse(int a, int n) {
        int x, y;
        extendedGCD(a, n, &x, &y);
        return x % n;
    }

    void extendedGCD(int a, int b, int* x, int* y) {
        if (b == 0) {
            *x = 1;
            *y = 0;
            return;
        }
        int d = a / b, xp, yp;
        extendedGCD(b, a % b, &xp, &yp);
        *x = yp;
        *y = xp - (d * yp);
    }

    float radicalInverse(int a) {
        int base = Primes[m_prime_index];
        const float inv_base = 1.f / (float) base;
        int reversed_digits = 0;
        float inv_base_N = 1;
        while (a) {
            int next = a / base;
            int digit = a - next * base;
            reversed_digits = reversed_digits * base + digit;
            inv_base_N *= inv_base;
            a = next;
        }
        return std::min(reversed_digits * inv_base_N, 1.f);
    }

    float scrambledRadicalInverse(int a, int* perm) {
        int base = Primes[m_prime_index];
        float inv_base = 1.f / (float) base;
        int reversed_digits = 0;
        float inv_base_N = 1;
        while (a) {
            int next = a / base;
            int digit = a - next * base;
            reversed_digits = reversed_digits * base + perm[digit];
            inv_base_N *= inv_base;
            a = next;
        }
        return std::min(inv_base_N * (reversed_digits + inv_base * perm[0] / (1.f - inv_base)), 1.f);
    }

public:
    Halton(const Properties &properties)
    : Sampler(properties) {
        if (m_rad_inv_perms.empty()) populateRadInvPerms();
    }

    void seed(int sampleIndex) override {
        m_prime_index = 0;
        m_mask = 0;
        m_sample_index = sampleIndex;
    }

    void seed(const Point2i &pixel, int sampleIndex) override {
        m_prime_index = 0;

        const uint64_t a = (uint64_t(pixel.x()) << 32) ^ pixel.y();
        m_pcg.seed(1337, a);
        m_mask = m_pcg.nextFloat();

        m_sample_index = sampleIndex;
    }

    float sampleDimension() {
        if (m_prime_index == 0) return radicalInverse(m_sample_index >> m_base_exps[0]);
        else if (m_prime_index == 1) return radicalInverse(m_sample_index / m_base_scales[1]);
        else {
            int* perm = &m_rad_inv_perms[Prime_Sums[m_prime_index]];
            return scrambledRadicalInverse(m_sample_index, perm);
        }
    }

    float next() override {
        auto result = sampleDimension();
        m_prime_index++;
        m_prime_index %= Primes.size();
        result += m_mask;
        if (result > 1.f) result -= 1.f;
        return result;
    }

    Point2 next2D() override {
        Point2 result;
        result[0] = next();
        result[1] = next();
        return result;
    }

    ref<Sampler> clone() const override {
        return std::make_shared<Halton>(*this);
    }

    std::string toString() const override {
        return tfm::format(
            "Halton[\n"
            "  count = %d\n"
            "]",
            m_samplesPerPixel
        );
    }
};

}

REGISTER_SAMPLER(Halton, "halton")
