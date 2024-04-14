#include "fresnel.hpp"
#include <lightwave.hpp>

namespace lightwave {

class Dielectric : public Bsdf {
    ref<Texture> m_ior;
    ref<Texture> m_reflectance;
    ref<Texture> m_transmittance;

public:
    Dielectric(const Properties &properties) {
        m_ior           = properties.get<Texture>("ior");
        m_reflectance   = properties.get<Texture>("reflectance");
        m_transmittance = properties.get<Texture>("transmittance");
    }

    Color albedo(const Point2 &uv) override {
        return m_reflectance->evaluate(uv);
    }

    BsdfEval evaluate(const Point2 &uv, const Vector &wo,
                      const Vector &wi) const override {
        // the probability of a light sample picking exactly the direction `wi'
        // that results from reflecting or refracting `wo' is zero, hence we can
        // just ignore that case and always return black
        return BsdfEval::invalid();
    }

    BsdfSample sample(const Point2 &uv, const Vector &wo,
                      Sampler &rng) const override {
        BsdfSample result;

        auto n = m_ior->scalar(uv);
        Vector normal(0,0,1);
        if (Frame::cosTheta(wo) < 0) {
            n = 1.f / n;
            normal = -normal;
        }

        auto fresnel = fresnelDielectric(Frame::cosTheta(wo), n);
        if (rng.next() < fresnel) {
            result.wi = reflect(wo, Vector(0,0,1));
            result.weight = m_reflectance->evaluate(uv);
        } else {
            result.wi = refract(wo, normal, n);
            result.weight = m_transmittance->evaluate(uv) / pow(n, 2);
        }

        return result;
    }

    std::string toString() const override {
        return tfm::format("Dielectric[\n"
                           "  ior           = %s,\n"
                           "  reflectance   = %s,\n"
                           "  transmittance = %s\n"
                           "]",
                           indent(m_ior), indent(m_reflectance),
                           indent(m_transmittance));
    }
};

} // namespace lightwave

REGISTER_BSDF(Dielectric, "dielectric")
