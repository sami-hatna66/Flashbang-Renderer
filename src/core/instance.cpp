#include <lightwave/core.hpp>
#include <lightwave/instance.hpp>
#include <lightwave/registry.hpp>
#include <lightwave/sampler.hpp>

namespace lightwave {

void Instance::transformFrame(SurfaceEvent &surf) const {
    surf.position = m_transform->apply(surf.position);

    if (m_normals) {
        auto normal_colour = m_normals->evaluate(surf.uv);
        Vector normal_M{2.f * normal_colour.r() - 1, 2.f * normal_colour.g() - 1, 2.f * normal_colour.b() - 1};
        Vector new_normal = normal_M.x() * surf.frame.tangent + normal_M.y() * surf.frame.bitangent + normal_M.z() * surf.frame.normal;
        surf.frame.normal = m_transform->applyNormal(new_normal);
        surf.frame = Frame(surf.frame.normal);
    } else {
        surf.frame.tangent = m_transform->apply(surf.frame.tangent);
        surf.frame.bitangent = m_transform->apply(surf.frame.bitangent);
        surf.frame.bitangent *= (m_flipNormal ? -1.f : 1.f);
        surf.frame.normal = surf.frame.tangent.cross(surf.frame.bitangent);
        surf.frame = Frame(surf.frame.normal);
    }
    surf.pdf /= surf.frame.tangent.cross(surf.frame.bitangent).length();

    surf.frame.tangent = surf.frame.tangent.normalized();
    surf.frame.bitangent = surf.frame.bitangent.normalized();
    surf.frame.normal = surf.frame.normal.normalized();
}

bool Instance::intersect(const Ray &worldRay, Intersection &its, Sampler &rng) const {
    if (m_alpha) its.alphaMask = m_alpha.get();
    else its.alphaMask = nullptr;

    its.opacity = m_opacity;

    if (!m_transform) {
        // fast path, if no transform is needed
        Ray localRay = worldRay;
        if (m_shape->intersect(localRay, its, rng)) {
            its.instance = this;
            return true;
        } else {
            return false;
        }
    }

    const float previousT = its.t;
    Ray localRay;

    localRay = m_transform->inverse(worldRay);

    // Save scale and transform its.t before normalizing localRay
    auto scale = localRay.direction.length();
    its.t *= scale;

    localRay = localRay.normalized();

    const bool wasIntersected = m_shape->intersect(localRay, its, rng);

    if (wasIntersected) {
        if (m_opacity != 1.f && rng.next() > m_opacity) {
            its.t = previousT;
            return false;
        }

        its.instance = this;

        transformFrame(its);

        // Undo its.t scaling
        its.t /= scale;
        return true;
    } else {
        its.t = previousT;
        return false;
    }
}

Bounds Instance::getBoundingBox() const {
    if (!m_transform) {
        // fast path
        return m_shape->getBoundingBox();
    }

    const Bounds untransformedAABB = m_shape->getBoundingBox();
    if (untransformedAABB.isUnbounded()) {
        return Bounds::full();
    }

    Bounds result;
    for (int point = 0; point < 8; point++) {
        Point p = untransformedAABB.min();
        for (int dim = 0; dim < p.Dimension; dim++) {
            if ((point >> dim) & 1) {
                p[dim] = untransformedAABB.max()[dim];
            }
        }
        p = m_transform->apply(p);
        result.extend(p);
    }
    return result;
}

Point Instance::getCentroid() const {
    if (!m_transform) {
        // fast path
        return m_shape->getCentroid();
    }

    return m_transform->apply(m_shape->getCentroid());
}

AreaSample Instance::sampleArea(Sampler &rng) const {
    AreaSample sample = m_shape->sampleArea(rng);
    transformFrame(sample);
    return sample;
}

}

REGISTER_CLASS(Instance, "instance", "default")
