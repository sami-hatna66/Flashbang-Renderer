#include <lightwave.hpp>
#include <random>

namespace lightwave {
class Sphere : public Shape {
private:
    Point2 uvFromPosition(const Point &p) const {
        return {0.5f + (atan2(p.z(), p.x()) / (2.f * Pi)), 0.5f + (asin(p.y()) / Pi)};
    }

    inline void populate(SurfaceEvent& surf, const Point& position) const {
        surf.position = position;

        surf.frame.normal = (Vector(surf.position.x(), surf.position.y(), surf.position.z()) - m_center / m_radius).normalized(); // normal from intersection
        surf.frame = Frame(surf.frame.normal);

        surf.uv = uvFromPosition(surf.position);

        surf.pdf = 1.f / (4.f * Pi);
    }

    Vector m_center;
    float m_radius;
public:
    Sphere(const Properties &properties) : m_center{0.f, 0.f, 0.f}, m_radius{1.f} {}

    bool intersect(const Ray &ray, Intersection &its, Sampler &rng) const override {
        // Vector from origin to center
        Vector o_2_c = m_center - ray.origin;
        // Project o_2_c onto ray direction
        float tca = o_2_c.dot(ray.direction);
        // If projection is negative, sphere is behind ray origin
        if (tca < 0.f) return false;

        // Avoid extraneous sqrt by keeping distance squared and comparing with radius squared
        float d2 = o_2_c.dot(o_2_c) - tca * tca;
        // If distance^2 is greater than r^2, ray goes past sphere without intersection
        if (d2 > m_radius * m_radius) return false;

        // Distance from projection point to intersection point
        float thc = sqrt((m_radius * m_radius) - d2);
        float t_min = tca - thc; // first intersection point
        float t_max = tca + thc; // second intersection point

        bool min_valid = t_min > Epsilon && t_min < its.t;
        if (min_valid && its.alphaMask) {
            Point2 uv_min = uvFromPosition(ray(t_min));
            min_valid = its.alphaMask->scalar(uv_min) > 0.5f;
        }

        bool max_valid = t_max > Epsilon && t_max < its.t;
        if (max_valid && its.alphaMask) {
            Point2 uv_max = uvFromPosition(ray(t_max));
            max_valid = its.alphaMask->scalar(uv_max) > 0.5f;
        }

        if (!min_valid && !max_valid) return false;

        its.t = min_valid ? t_min : t_max;

        Ray norm_ray = ray.normalized();
        const Point position = norm_ray(its.t);
 
        populate(its, position);

        return true;
    }

    Bounds getBoundingBox() const override { 
        return Bounds(Point{-1.f, -1.f, -1.f}, Point{1.f, 1.f, 1.f});
    }

    Point getCentroid() const override {
        return Point(0.f);
    }

    AreaSample sampleArea(Sampler &rng) const override {
        auto position = squareToUniformSphere(rng.next2D());
        AreaSample sample;
        populate(sample, position);
        return sample;
    }

    std::string toString() const override { 
        return "Sphere[]";
    } 
};
}

REGISTER_SHAPE(Sphere, "sphere")
