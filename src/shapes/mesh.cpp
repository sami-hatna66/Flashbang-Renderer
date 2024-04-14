#include <lightwave.hpp>

#include "../core/plyparser.hpp"
#include "accel.hpp"

namespace lightwave {

/**
 * @brief A shape consisting of many (potentially millions) of triangles, which share an index and vertex buffer.
 * Since individual triangles are rarely needed (and would pose an excessive amount of overhead), collections of
 * triangles are combined in a single shape.
 */
class TriangleMesh : public AccelerationStructure {
    /**
     * @brief The index buffer of the triangles.
     * The n-th element corresponds to the n-th triangle, and each component of the element corresponds to one
     * vertex index (into @c m_vertices ) of the triangle.
     * This list will always contain as many elements as there are triangles.
     */
    std::vector<Vector3i> m_triangles;
    /**
     * @brief The vertex buffer of the triangles, indexed by m_triangles.
     * Note that multiple triangles can share vertices, hence there can also be fewer than @code 3 * numTriangles @endcode
     * vertices.
     */
    std::vector<Vertex> m_vertices;
    /// @brief The file this mesh was loaded from, for logging and debugging purposes.
    std::filesystem::path m_originalPath;
    /// @brief Whether to interpolate the normals from m_vertices, or report the geometric normal instead.
    bool m_smoothNormals;

protected:
    int numberOfPrimitives() const override {
        return int(m_triangles.size());
    }

    bool intersect(int primitiveIndex, const Ray &ray, Intersection &its, Sampler &rng) const override {
        auto& triangle = m_triangles[primitiveIndex];
        auto& v0 = m_vertices[triangle.x()];
        auto& v1 = m_vertices[triangle.y()];
        auto& v2 = m_vertices[triangle.z()];

        // Möller-Trumbore algorithm
        static constexpr float Epsilon1 = 1e-4f;
        static constexpr float Epsilon2 = 1e-8f;

        // Compute vectors for two triangle edges
        Vector e1 = v1.position - v0.position;
        Vector e2 = v2.position - v0.position;
        // Vector perpendicular to ray and triangle edge (normal to plane)
        // Used to define plane triangle lies on
        Vector pvec = ray.direction.cross(e2);

        // Edge and plane normal dot product is 0 if ray is parallel to triangle
        // Therefore, no intersection
        float det = e1.dot(pvec);
        if (det > -Epsilon2 && det < Epsilon2) return false;

        float invDet = 1.f / det;

        // Compute first barycentric coordinate 
        Vector tvec = ray.origin - v0.position;
        float u = tvec.dot(pvec) * invDet;
        if (u < 0.f || u > 1.f) return false;

        // Compute second barycentric coordinate
        Vector qvec = tvec.cross(e1);
        float v = ray.direction.dot(qvec) * invDet;
        if (v < 0.f || u + v > 1.f) return false;
        
        // Intersection distance along the ray
        float t = e2.dot(qvec) * invDet;

        // Make sure current intersection is the closest
        if (t > Epsilon1 && t < its.t) {
            auto uv = Vertex::interpolate({u,v}, v0, v1, v2).texcoords;

            if (its.alphaMask && its.alphaMask->scalar(uv) < 0.5f) return false;

            its.uv = uv;
            its.t = t;
            its.position = ray(its.t);
            
            its.pdf = 0.f;
            if (m_smoothNormals) {
                // Gouraud shading using vertex normals
                auto& n0 = v0.normal;
                auto& n1 = v1.normal; 
                auto& n2 = v2.normal;
                its.frame.normal = Vector(interpolateBarycentric({u, v}, n0.x(), n1.x(), n2.x()),
                                          interpolateBarycentric({u, v}, n0.y(), n1.y(), n2.y()),
                                          interpolateBarycentric({u, v}, n0.z(), n1.z(), n2.z())).normalized(); 
            } else {
                // Compute face normal if smooth normals aren't wanted
                its.frame.normal = (v1.position - v0.position).cross(v2.position - v0.position).normalized();
            }
            its.frame = Frame(its.frame.normal);
            return true;
        }
        else return false;
    }

    Bounds getBoundingBox(int primitiveIndex) const override {
        Bounds result;

        auto& firstVertex = m_vertices[m_triangles[primitiveIndex].x()].position;
        auto& secondVertex = m_vertices[m_triangles[primitiveIndex].y()].position;
        auto& thirdVertex = m_vertices[m_triangles[primitiveIndex].z()].position;

        result.extend(firstVertex);
        result.extend(secondVertex);
        result.extend(thirdVertex);

        return result;
    }

    Point getCentroid(int primitiveIndex) const override {
        // Compute centroid as center of bounding box
        return getBoundingBox(primitiveIndex).center();
    }

public:
    TriangleMesh(const Properties &properties) {
        m_originalPath = properties.get<std::filesystem::path>("filename");
        m_smoothNormals = properties.get<bool>("smooth", true);
        readPLY(m_originalPath.string(), m_triangles, m_vertices);
        logger(EInfo, "loaded ply with %d triangles, %d vertices",
            m_triangles.size(),
            m_vertices.size()
        );

        buildAccelerationStructure();
    }

    AreaSample sampleArea(Sampler &rng) const override {
        // only implement this if you need triangle mesh area light sampling for your rendering competition
        NOT_IMPLEMENTED
    }

    std::string toString() const override {
        return tfm::format(
            "Mesh[\n"
            "  vertices = %d,\n"
            "  triangles = %d,\n"
            "  filename = \"%s\"\n"
            "]",
            m_vertices.size(),
            m_triangles.size(),
            m_originalPath.generic_string()
        );
    }
};

}

REGISTER_SHAPE(TriangleMesh, "mesh")
