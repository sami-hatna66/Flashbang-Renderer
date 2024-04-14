#include "fresnel.hpp"
#include "microfacet.hpp"
#include <lightwave.hpp>

namespace lightwave {

class RoughConductor : public Bsdf {
    ref<Texture> m_reflectance;
    ref<Texture> m_roughness;

public:
    RoughConductor(const Properties &properties) {
        m_reflectance = properties.get<Texture>("reflectance");
        m_roughness   = properties.get<Texture>("roughness");
    }

    Color albedo(const Point2 &uv) override {
        return m_reflectance->evaluate(uv);
    }

    BsdfEval evaluate(const Point2 &uv, const Vector &wo,
                      const Vector &wi) const override {
        // Using the squared roughness parameter results in a more gradual
        // transition from specular to rough. For numerical stability, we avoid
        // extremely specular distributions (alpha values below 10^-3)
        const auto alpha = std::max(float(1e-3), sqr(m_roughness->scalar(uv)));

        BsdfEval result;

        auto wm = (wi + wo).normalized();
        auto R = m_reflectance->evaluate(uv);
        auto D = microfacet::evaluateGGX(alpha, wm);
        auto g1wi = microfacet::smithG1(alpha, wm, wi);
        auto g1wo = microfacet::smithG1(alpha, wm, wo);

        result.value = (R * D * g1wi * g1wo) / abs(4.f * Frame::cosTheta(wo));

        return result;
    }

    BsdfSample sample(const Point2 &uv, const Vector &wo,
                      Sampler &rng) const override {
        const auto alpha = std::max(float(1e-3), sqr(m_roughness->scalar(uv)));

        BsdfSample result;

        auto wm = microfacet::sampleGGXVNDF(alpha, wo, rng.next2D());
        result.wi = reflect(wo, wm);

        auto R = m_reflectance->evaluate(uv);
        auto g1wi = microfacet::smithG1(alpha, wm, result.wi);

        result.weight = R * g1wi;

        return result;
    }

    std::string toString() const override {
        return tfm::format("RoughConductor[\n"
                           "  reflectance = %s,\n"
                           "  roughness = %s\n"
                           "]",
                           indent(m_reflectance), indent(m_roughness));
    }
};

} // namespace lightwave

REGISTER_BSDF(RoughConductor, "roughconductor")
