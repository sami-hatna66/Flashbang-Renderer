#include <lightwave.hpp>

namespace lightwave {

class AlbedoIntegrator : public SamplingIntegrator {

public:
    AlbedoIntegrator(const Properties &properties)
    : SamplingIntegrator(properties) {}

    Color Li(const Ray &ray, Sampler &rng) override {
        auto its = m_scene->intersect(ray, rng);
        if (its && its.instance->bsdf()) return its.instance->bsdf()->albedo(its.uv);
        else return Color(0);
    }

    std::string toString() const override {
        return tfm::format(
            "AlbedoIntegrator"
        );
    }
};

}

REGISTER_INTEGRATOR(AlbedoIntegrator, "albedo")
