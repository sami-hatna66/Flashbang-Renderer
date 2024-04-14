#include <lightwave.hpp>

namespace lightwave {

class NormalIntegrator : public SamplingIntegrator {
    bool m_remap;

public:
    NormalIntegrator(const Properties &properties)
    : SamplingIntegrator(properties) {
        m_remap = properties.get<bool>("remap", true);
    }

    Color Li(const Ray &ray, Sampler &rng) override {
        // Intersect ray with scene
        auto its = m_scene->intersect(ray, rng);
        
        // Remap normals [-1,1] -> [0,1]
        if (m_remap) {
            its.frame.normal = Vector(
                (its.frame.normal.x() + 1.f) / 2.f,
                (its.frame.normal.y() + 1.f) / 2.f, 
                (its.frame.normal.z() + 1.f) / 2.f
            );
        }
        return Color(its.frame.normal);
    }

    std::string toString() const override {
        return tfm::format(
            "NormalIntegrator[\n"
            "  remap = %s,\n"
            "]",
            indent(m_remap)
        );
    }
};

}

REGISTER_INTEGRATOR(NormalIntegrator, "normals")
