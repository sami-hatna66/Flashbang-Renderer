/**
 * @file math.hpp
 * @brief Contains all geometrical constructs (Points, Matrices, Rays, etc), as well as commonly used mathematical constants and functions.
 */

#pragma once

#include <lightwave/core.hpp>
#include <lightwave/logger.hpp>

#include <cmath>
#include <algorithm>
#include <array>
#include <optional>

namespace lightwave {

// MARK: - useful constants

/// @brief Pi
static constexpr float Pi     = 3.14159265358979323846f;
/// @brief 1 / Pi
static constexpr float InvPi  = 0.31830988618379067154f;
/// @brief 1 / (2 * Pi)
static constexpr float Inv2Pi = 0.15915494309189533577f;
/// @brief 1 / (4 * Pi)
static constexpr float Inv4Pi = 0.07957747154594766788f;
/// @brief Pi / 2
static constexpr float Pi2    = 1.57079632679489661923f;
/// @brief Pi / 4
static constexpr float Pi4    = 0.78539816339744830961f;
/// @brief sqrt(2)
static constexpr float Sqrt2  = 1.41421356237309504880f;

/// @brief Multiply by this constant to convert degrees into radians.
static constexpr float Deg2Rad = Pi / 180.0f;
/// @brief Multiply by this constant to convert radians into degrees.
static constexpr float Rad2Deg = 180.0f * InvPi;

/**
 * @brief The tolerance threshold for floating point inaccuracies.
 * @example When starting a ray at a surface, it can happen that the surface reports a self-intersection.
 * Mathematically, the intersection distance should be zero, allowing us to discard the intersection, but in reality
 * floating point inaccuracies can report larger distances that would be considered valid intersections.
 * This threshold allows us to solve this (and many other) inaccuracy problem, by treating anything below this threshold
 * as zero.
 */
static constexpr float Epsilon = 1e-5f;

/// @brief Infinity
static constexpr float Infinity = std::numeric_limits<float>::infinity();

// MARK: - utility functions

/// @brief Square root function.
inline float sqrt(float v) { return std::sqrt(v); }
/// @brief Square function, i.e., @code sqr(v) = v * v @endcode.
inline float sqr(float v) { return v * v; }

/// @brief Maximum of two numbers.
inline float max(float a, float b) { return std::max(a, b); }
/// @brief Minimum of two numbers.
inline float min(float a, float b) { return std::min(a, b); }

/// @brief Returns a value of @c mag with the sign of @c sgn .
inline float copysign(float mag, float sgn) { return std::copysign(mag, sgn); }
/// @brief Returns the absolute value of @c v .
inline float abs(float v) { return std::abs(v); }

/// @brief Clamps a value @c v to lie in the interval from @c lo to @c hi .
inline float clamp(float v, float lo, float hi) { return max(lo, min(v, hi)); }
/// @brief Clamps a value @c v to lie in the unit interval.
inline float saturate(float v) { return clamp(v, 0, 1); }

/**
 * @brief Safe square root, which clamps negative input values to zero.
 * @note Use this when floating point errors need to be avoided, e.g., for @code sin = safe_sqrt(1 - sqr(cos)) @endcode.
 */
inline float safe_sqrt(float v) { return v <= 0 ? 0 : std::sqrt(v); }
/**
 * @brief Safe arcus cosine function, which clamps the input to -1 to +1.
 * @note Use this when floating point errors need to be avoided.
 */
inline float safe_acos(float v) { return std::acos(clamp(v, -1, +1)); }

// MARK: - points and vectors

#define BUILD1(expr) \
    result; \
    for (int i = 0; i < result.Dimension; i++) result[i] = expr; \
    return result;

/// @brief A point in @c D dimensions, the components of which are stored with datatype @c T .
template<typename T, int D>
class TPoint {
protected:
    /// @brief The components that constitute the point.
    std::array<T, D> m_data;

public:
    /// @brief The datatype used to store the components of the point.
    typedef T Type;
    /// @brief The dimensionality of the point.
    static constexpr int Dimension = D;

    /// @brief Constructs a point at the origin.
    TPoint() { std::fill(m_data.begin(), m_data.end(), Type(0)); }
    /// @brief Constructs a point from a given array.
    TPoint(const std::array<Type, Dimension> &data) : m_data(data) {}
    /// @brief Constructs a point from one of dimensionality one less, and a value for the last dimension.
    TPoint(const TPoint<T, D - 1> &other, const T &v) {
        std::copy(other.data().begin(), other.data().end(), m_data.begin());
        m_data.back() = v;
    }
    /// @brief Constructs a point whose components all have the value @c v .
    explicit TPoint(const Type &v) { std::fill(m_data.begin(), m_data.end(), v); }

    /// @brief Constructs a two-dimensional point.
    TPoint(const Type &x, const Type &y) : m_data({ x, y }) {}
    /// @brief Constructs a three-dimensional point.
    TPoint(const Type &x, const Type &y, const Type &z) : m_data({ x, y, z }) {}
    /// @brief Constructs a four-dimensional point (used for homogeneous coordinates).
    TPoint(const Type &x, const Type &y, const Type &z, const Type &w) : m_data({ x, y, z, w }) {}

    /// @brief Returns an array of the components of this point.
    const std::array<Type, Dimension> &data() const { return m_data; }
    /// @brief Returns an array of the components of this point that can be modified.
    std::array<Type, Dimension> &data() { return m_data; }
    
    /// @brief Access a component of this point, with an index ranging from @c 0 to @code Dimension - 1 @endcode .
    const Type &operator[](int i) const { return m_data[i]; }
    /// @brief Access a component of this point that can be modified, with an index ranging from @c 0 to @code Dimension - 1 @endcode .
    Type &operator[](int i) { return m_data[i]; }

    const Type &x() const { static_assert(Dimension >= 1); return m_data[0]; }
    const Type &y() const { static_assert(Dimension >= 2); return m_data[1]; }
    const Type &z() const { static_assert(Dimension >= 3); return m_data[2]; }
    const Type &w() const { static_assert(Dimension >= 4); return m_data[3]; }

    Type &x() { static_assert(Dimension >= 1); return m_data[0]; }
    Type &y() { static_assert(Dimension >= 2); return m_data[1]; }
    Type &z() { static_assert(Dimension >= 3); return m_data[2]; }
    Type &w() { static_assert(Dimension >= 4); return m_data[3]; }

    /// @brief Returns the elementwise minimum of two points.
    friend auto elementwiseMin(const TPoint &a, const TPoint &b) { TPoint BUILD1(std::min(a[i], b[i])) }
    /// @brief Returns the elementwise maximum of two points.
    friend auto elementwiseMax(const TPoint &a, const TPoint &b) { TPoint BUILD1(std::max(a[i], b[i])) }

    /// @brief Converts the point to a different datatype.
    template<typename Type2>
    auto cast() const { TPoint<Type2, Dimension> BUILD1(Type2(m_data[i])) }

    /// @brief Checks whether two points are exactly identical.
    bool operator==(const TPoint &other) const { return m_data == other.m_data; }
    /// @brief Checks whether two points are not exactly identical.
    bool operator!=(const TPoint &other) const { return m_data != other.m_data; }

    /// @brief Returns whether the point lies at the origin, i.e., all components are zero. 
    bool isZero() const {
        return std::all_of(m_data.begin(), m_data.end(), [](const Type &v) { return v == 0; });
    }
};

/// @brief A vector in @c D dimensions, the components of which are stored with datatype @c T .
template<typename Type, int Dimension>
class TVector : public TPoint<Type, Dimension> {
public:
    using TPoint<Type, Dimension>::TPoint;
    using TPoint<Type, Dimension>::m_data;

    explicit TVector(const TPoint<Type, Dimension> &point)
    : TPoint<Type, Dimension>(point.data()) {}

    /// @brief Computes the dot product (aka scalar product) with respect to another vector.
    float dot(const TVector &other) const {
        float result = 0;
        for (int i = 0; i < Dimension; i++) result += m_data[i] * other.m_data[i];
        return result;
    }

    /// @brief Computes the cross product with respect to another vector.
    TVector cross(const TVector &other) const {
        static_assert(Dimension == 3);
        return {
            this->y() * other.z() - this->z() * other.y(),
            this->z() * other.x() - this->x() * other.z(),
            this->x() * other.y() - this->y() * other.x()
        };
    }

    /// @brief Computes the squared length of this vector. 
    float lengthSquared() const { return dot(*this); }
    /// @brief Computes the length of this vector. 
    float length() const { return std::sqrt(lengthSquared()); }
    /// @brief Returns a normalized copy of this vector.
    auto normalized() const { return *this / length(); }

    /// @brief Returns the length of this vector along with a normalized copy.
    auto lengthAndNormalized() const {
        const float length = this->length();
        return std::make_pair(length, *this / length);
    }

    friend auto operator-(const TVector &a) { TVector BUILD1(-a[i]) }
    friend auto operator*(const Type &a, const TVector &b) { TVector BUILD1(a * b[i]) }
    friend auto operator*(const TVector &a, Type b) { TVector BUILD1(a[i] * b) }
    friend auto operator/(const TVector &a, Type b) { TVector BUILD1(a[i] / b) }
    friend auto operator+(const TVector &a, const TVector &b) { TVector BUILD1(a[i] + b[i]) }
    friend auto operator-(const TVector &a, const TVector &b) { TVector BUILD1(a[i] - b[i]) }
    friend auto operator*(const TVector &a, const TVector &b) { TVector BUILD1(a[i] * b[i]) }
    friend auto operator/(const TVector &a, const TVector &b) { TVector BUILD1(a[i] / b[i]) }
    
    /// @brief Returns the elementwise minimum of two vectors.
    friend auto elementwiseMin(const TVector &a, const TVector &b) { TVector BUILD1(std::min(a[i], b[i])) }
    /// @brief Returns the elementwise maximum of two vectors.
    friend auto elementwiseMax(const TVector &a, const TVector &b) { TVector BUILD1(std::max(a[i], b[i])) }

    /// @brief Returns the lowest component of the vector.
    auto minComponent() const { return *std::min_element(m_data.begin(), m_data.end()); }
    /// @brief Returns the highest component of the vector.
    auto maxComponent() const { return *std::max_element(m_data.begin(), m_data.end()); }

    /// @brief Returns the index of the lowest component of the vector.
    auto minComponentIndex() const { return (int)std::distance(m_data.begin(), std::min_element(m_data.begin(), m_data.end())); }
    /// @brief Returns the index of the highest component of the vector.
    auto maxComponentIndex() const { return (int)std::distance(m_data.begin(), std::max_element(m_data.begin(), m_data.end())); }

    /// @brief Returns a sum of all components of the vector.
    auto sum() const { Type value { 0 }; for (int i = 0; i < Dimension; i++) value += m_data[i]; return value; }
    /// @brief Returns a product of all components of the vector.
    auto product() const { Type value { 1 }; for (int i = 0; i < Dimension; i++) value *= m_data[i]; return value; }

    auto operator*=(const Type &other) { return *this = *this * other; }
    auto operator/=(const Type &other) { return *this = *this / other; }
    auto operator+=(const TVector &other) { return *this = *this + other; }
    auto operator-=(const TVector &other) { return *this = *this - other; }
    auto operator*=(const TVector &other) { return *this = *this * other; }
    auto operator/=(const TVector &other) { return *this = *this / other; }

    /// @brief Converts the vector to a different datatype.
    template<typename Type2>
    auto cast() const { TVector<Type2, Dimension> BUILD1(Type2(m_data[i])) }
};

#undef BUILD_VECTOR

template<typename Type, int Dimension>
auto operator+(const TPoint<Type, Dimension> &a, const TVector<Type, Dimension> &b) {
    TPoint<Type, Dimension> BUILD1(a[i] + b[i])
}

template<typename Type, int Dimension>
auto operator+=(TPoint<Type, Dimension> &a, const TVector<Type, Dimension> &b) {
    a = a + b;
}

template<typename Type, int Dimension>
auto operator-(const TPoint<Type, Dimension> &a, const TVector<Type, Dimension> &b) {
    TPoint<Type, Dimension> BUILD1(a[i] - b[i])
}

template<typename Type, int Dimension>
auto operator-=(TPoint<Type, Dimension> &a, const TVector<Type, Dimension> &b) {
    a = a - b;
}

template<typename Type, int Dimension>
auto operator-(const TPoint<Type, Dimension> &a, const TPoint<Type, Dimension> &b) {
    TVector<Type, Dimension> BUILD1(a[i] - b[i])
}

#define BUILD2(expr) \
    result; \
    for (int row = 0; row < result.Rows; row++) { \
        for (int column = 0; column < result.Columns; column++) { \
            result(row, column) = expr; \
        } \
    } \
    return result;

// MARK: - matrix

/// @brief A matrix with @c R rows and @c C columns, the elements of which are stored with datatype @c T .
template<typename T, int R, int C>
class TMatrix {
    /// @brief The elements that constitute the matrix, in row-major format.
    std::array<std::array<T, C>, R> m_data;

public:
    /// @brief The datatype used to store the elements of the point.
    typedef T Type;
    /// @brief The number of rows in the matrix.
    static constexpr int Rows = R;
    /// @brief The number of columns in the matrix.
    static constexpr int Columns = C;
    
    /// @brief Creates a matrix filled with zeros.
    TMatrix() {}
 
    TMatrix(std::initializer_list<Type> l) {
        assert(l.size() == (Rows * Columns)); // "cannot initialize Matrix<%d, %d> with %d elements", Rows, Columns, l.size()

        auto it = l.begin();
        for (int row = 0; row < Rows; row++) {
            for (int column = 0; column < Columns; column++) {
                (*this)(row, column) = *it++;
            }
        }
    }

    /// @brief Returns an element of this matrix.
    const Type &operator()(int row, int column) const { return m_data[row][column]; }
    /// @brief Returns a reference to an element of this matrix.
    Type &operator()(int row, int column) { return m_data[row][column]; }

    /// @brief Returns a row of the matrix as vector, with index ranging from @c 0 to @code Rows - 1 @endcode .
    auto row(int rowIndex) const { TVector<Type, Columns> BUILD1((*this)(rowIndex, i)) }
    /// @brief Returns a column of the matrix as vector, with index ranging from @c 0 to @code Columns - 1 @endcode .
    auto column(int columnIndex) const { TVector<Type, Rows> BUILD1((*this)(i, columnIndex)) }

    /// @brief Sets a row of the matrix to a given vector, with index ranging from @c 0 to @code Rows - 1 @endcode .
    auto setRow(int rowIndex, const TVector<Type, Columns> &vector) {
        for (int c = 0; c < Columns; c++) (*this)(rowIndex, c) = vector[c];
    }
    /// @brief Sets a column of the matrix to a given vector, with index ranging from @c 0 to @code Columns - 1 @endcode .
    auto setColumn(int columnIndex, const TVector<Type, Rows> &vector) {
        for (int r = 0; r < Rows; r++) (*this)(r, columnIndex) = vector[r];
    }

    /// @brief Returns the product of @code matrix * vector @endcode .
    auto operator*(const TVector<Type, Columns> &vector) const {
        TVector<Type, Rows> BUILD1(row(i).dot(vector))
    }
    /// @brief Returns the product of @code matrix * otherMatrix @endcode .
    template<int Columns2>
    auto operator*(const TMatrix<Type, Columns, Columns2> &other) const {
        TMatrix<Type, Rows, Columns2> BUILD2((*this).row(row).dot(other.column(column)))
    }

    /// @brief Returns the transpose of this matrix. 
    auto transpose() const {
        TMatrix<Type, Columns, Rows> BUILD2((*this)(column, row))
    }

    /// @brief Returns a slice of this matrix of size @c Rows2 x @c Columns2 , offset by @c startRow and @c startColumn .
    template<int Rows2, int Columns2>
    auto submatrix(int startRow, int startColumn) const {
        assert(startRow >= 0 && startRow + Rows2 <= Rows);
        assert(startColumn >= 0 && startColumn + Columns2 <= Columns);
        TMatrix<Type, Rows2, Columns2> BUILD2((*this)(startRow + row, startColumn + column))
    }

    /// @brief Returns the determinant of this matrix. 
    auto determinant() const {
        static_assert(Rows == Columns);
        static_assert(Rows >= 1);
        static_assert(Rows <= 3);

        if constexpr (Rows == 3 && Columns == 3) {
            float v = 0;
            for (int i = 0; i < 3; i++) {
                float a = 1, b = 1;
                for (int j = 0; j < 3; j++) {
                    a *= (*this)((i + j) % 3, j);
                    b *= (*this)((i + j) % 3, 2 - j);
                }
                v += a - b;
            }
            return v;
        } else if constexpr (Rows == 2 && Columns == 2) {
            return (*this)(0, 0) * (*this)(1, 1) - (*this)(0, 1) * (*this)(1, 0);
        } else if constexpr (Rows == 1 && Columns == 1) {
            return (*this)(0, 0);
        }
    }

    friend auto operator-(const TMatrix &a) { TMatrix BUILD2(-a(row, column)) }
    friend auto operator*(const Type &a, const TMatrix &b) { TMatrix BUILD2(a * b(row, column)) }
    friend auto operator*(const TMatrix &a, Type b) { TMatrix BUILD2(a(row, column) * b) }
    friend auto operator+(const TMatrix &a, const TMatrix &b) { TMatrix BUILD2(a(row, column) + b(row, column)) }
    friend auto operator-(const TMatrix &a, const TMatrix &b) { TMatrix BUILD2(a(row, column) - b(row, column)) }

    /// @brief Constructs an identity matrix for the given dimensions.
    static auto identity() { TMatrix BUILD2(row == column ? 1.0f : 0.0f) }
};

/**
 * @brief Bounds describe the range from one point to another (e.g., an axis-aligned bounding box).
 * In one dimension, this corresponds to an interval from one value to another.
 * In two dimensions, this corresponds to a rectangle.
 */
template<typename T, int D>
class TBounds {
public:
    using Point = TPoint<T, D>;
    using Vector = TVector<T, D>;

private:
    /// @brief The lower corner of the bounding box.
    Point m_min;
    /// @brief The upper corner of the bounding box.
    Point m_max;

public:
    /// @brief The datatype used to store the components of the point.
    typedef T Type;
    /// @brief The dimensionality of the point.
    static constexpr int Dimension = D;

    /// @brief Constructs an empty bounding box.
    static TBounds empty() { return TBounds(); }
    /// @brief Constructs a bounding box spanning all of space. 
    static TBounds full() {
        TBounds result;
        result.m_min = Point(-std::numeric_limits<T>::infinity());
        result.m_max = Point(+std::numeric_limits<T>::infinity());
        return result;
    }

    /// @brief Constructs an empty bounding box.
    TBounds()
    : m_min(+std::numeric_limits<T>::infinity()), m_max(-std::numeric_limits<T>::infinity()) {}

    TBounds(const Point &min, const Point &max)
    : m_min(min), m_max(max) {}

    /// @brief Extends this bounding box to also cover the region of another bounding box.
    void extend(const TBounds &other) {
        m_min = elementwiseMin(m_min, other.m_min);
        m_max = elementwiseMax(m_max, other.m_max);
    }

    /// @brief Extends this bounding box to contain a given point.
    void extend(const Point &point) {
        m_min = elementwiseMin(m_min, point);
        m_max = elementwiseMax(m_max, point);
    }

    /// @brief Returns whether the bounding box is empty.
    bool isEmpty() const {
        for (int dim = 0; dim < Dimension; dim++) {
            if (m_min[dim] >= m_max[dim]) {
                return true;
            }
        }
        return false;
    }

    /**
     * @brief Returns whether the bounding box covers an infinite amount of space.
     * @note It suffices if a single axis of the bounding box covers an infinite range.
     */
    bool isUnbounded() const {
        for (int dim = 0; dim < Dimension; dim++) {
            if (m_min[dim] < 0 && std::isinf(m_min[dim])) return true;
            if (m_max[dim] > 0 && std::isinf(m_max[dim])) return true;
        }
        return false;
    }

    /// @brief Clamps the components of a point so that it lies within this bounding box.
    Point clip(const Point &point) const {
        return elementwiseMax(m_min, elementwiseMin(point, m_max));
    }

    /// @brief Clamps the components of another bounding box so that it lies within this bounding box.
    TBounds clip(const TBounds &other) const {
        TBounds result;
        result.m_min = clip(other.m_min);
        result.m_max = clip(other.m_max);
        return result;
    }

    /// @brief Returns a shifted version of this bounding box by some given vector.
    TBounds operator+(const Vector &shift) const {
        return { m_min + shift, m_max + shift };
    }

    /// @brief Helper to iterate over the elements of a bounding box.
    struct iterator {
        iterator(const TBounds &bounds, const Point &value)
        : m_value(value), m_start(bounds.min()), m_end(bounds.max()) {}

        Point operator*() const { return m_value; }
        bool operator!=(const iterator &other) const { return m_value != other.m_value; }
        iterator &operator++() {
            for (int i = 0; i < Dimension; i++) {
                if (++m_value[i] < m_end[i]) break;
                if (i == Dimension - 1) break;
                m_value[i] = m_start[i];
            }
            return *this;
        }
    
    private:
        Point m_value;
        Point m_start, m_end;
    };

    /// @brief Returns the lower corner of this bounding box. 
    const Point &min() const { return m_min; }
    /// @brief Returns the upper corner of this bounding box. 
    const Point &max() const { return m_max; }
    /// @brief Returns a reference to the lower corner of this bounding box. 
    Point &min() { return m_min; }
    /// @brief Returns a reference to the upper corner of this bounding box. 
    Point &max() { return m_max; }

    /// @brief Returns the diagonal of the bounding box, i.e., @code max - min @endcode .
    Vector diagonal() const { return m_max - m_min; }
    /// @brief Returns the point that lies in the center of the bounding box. 
    Point center() const { return m_min + diagonal() / 2; }

    iterator begin() const { return { *this, m_min }; }
    iterator end() const {
        Point end = m_min;
        end[Dimension - 1] = m_max[Dimension - 1];
        return { *this, end };
    }

    /// @brief Tests whether the given point lies within this bounding box.
    bool includes(const Point &point) const {
        for (int dim = 0; dim < D; dim++) {
            if (point[dim] < m_min[dim]) return false;
            if (point[dim] > m_max[dim]) return false;
        }
        return true;
    }
};

/// @brief A two-dimensional point with floating point components.
using Point2 = TPoint<float, 2>;
/// @brief A two-dimensional point with integer components.
using Point2i = TPoint<int, 2>;
/// @brief A three-dimensional point with floating point components.
using Point = TPoint<float, 3>;

/// @brief A two-dimensional vector with floating point components.
using Vector2 = TVector<float, 2>;
/// @brief A two-dimensional vector with integer components.
using Vector2i = TVector<int, 2>;
/// @brief A three-dimensional vector with floating point components.
using Vector = TVector<float, 3>;
/// @brief A three-dimensional vector with integer components.
using Vector3i = TVector<int, 3>;
/// @brief A four-dimensional vector with floating point components (used for homogeneous coordinates).
using Vector4 = TVector<float, 4>;

/// @brief An integer rectangle (e.g., to describe the blocks of an image).
using Bounds2i = TBounds<int, 2>;
/// @brief A three-dimensional axis-aligned bounding box with floating point components.
using Bounds = TBounds<float, 3>;

/// @brief A 3x3 matrix with floating point components.
using Matrix3x3 = TMatrix<float, 3, 3>;
/// @brief A 4x4 matrix with floating point components (used for homogeneous coordinates).
using Matrix4x4 = TMatrix<float, 4, 4>;

/// @brief Computes the component-wise modulo operation of two points. 
template<int Dimension>
auto operator%(const TPoint<int, Dimension> &a, const TPoint<int, Dimension> &b) { TPoint<int, Dimension> BUILD1(a[i] % b[i]) }

/// @brief Computes the component-wise modulo operation of two points. 
template<int Dimension>
auto operator%(const TPoint<float, Dimension> &a, const TPoint<float, Dimension> &b) { TPoint<float, Dimension> BUILD1(fmodf(a[i], b[i])) }

#undef BUILD1
#undef BUILD2

/// @brief Computes the inverse of a 4x4 matrix. 
std::optional<Matrix4x4> invert(const Matrix4x4 &matrix);

/// @brief Builds an orthonormal basis from the given vectors. 
void buildOrthonormalBasis(const Vector &normal, Vector &tangent, Vector &bitangent);

// MARK: - rays and related structures

/**
 * @brief Reflects a vector @c w at a surface with normal @c n .
 * @note All vectors are assumed to point away from the surface.
 */
inline Vector reflect(const Vector &w, const Vector &n) {
    return 2 * n.dot(w) * n - w;
}

/**
 * @brief Refracts a vector @c w at a surface with normal @c n and relative index of refraction @c eta .
 * @note All vectors are assumed to point away from the surface.
 * @return Refracted vector, or zero vector in case of total internal reflection.
*/
inline Vector refract(const Vector &w, const Vector &n, float eta) {
    const float invEta = 1 / eta;
    const float k = 1 - sqr(invEta) * (1 - sqr(n.dot(w)));
    if (k < 0) {
        // total internal reflection
        return Vector();
    }
    return (invEta * n.dot(w) - sqrt(k)) * n - invEta * w;
}

/// @brief Describes a ray that propagates through space.
struct Ray {
    /// @brief The origin whether the ray starts (t = 0).
    Point origin;
    /// @brief The direction of the ray, which must always be normalized.
    Vector direction;
    /// @brief The number of bounces encountered by the ray, for use in integrators.
    int depth = 0;

    Ray() {}
    Ray(Point origin, Vector direction, int depth = 0)
    : origin(origin), direction(direction), depth(depth) {}

    /// @brief Computes a point on the ray for a given distance @c t .
    Point operator()(float t) const {
        return origin + t * direction;
    }

    /// @brief Returns a copy of the ray with normalized direction vector (useful after applying transforms). 
    Ray normalized() const {
        return Ray(
            origin,
            direction.normalized(),
            depth
        );
    }
};

/**
 * @brief Defines shading frames and common trigonometrical functions used within them.
 * In lightwave, we follow the convention that material functions (sampling and evaluation of Bsdfs and Emissions) happen
 * within a local frame where the normal vector is always [0,0,1], which simplifies much of their implementation.
 * This class defines the orientation of a surface, which allows us to transform world space vectors (e.g., ray directions)
 * into this local shading frame.
 * @warning Make sure that the vectors that constitute your Frame are always orthonormal and form a right-handed coordinate system.
 */
struct Frame {
    /// @brief The normal vector of the surface (implicitly given by the cross product of tangent and bitangent).
    Vector normal;
    /// @brief The tangent vector, which lies in the surface.
    Vector tangent;
    /// @brief The bitangent vector, which lies in the surface and is orthogonal to the tangent vector.
    Vector bitangent;

    Frame() {}

    /// @brief Constructs a frame with arbitrary tangent and bitangent vector from a given normal vector. 
    explicit Frame(const Vector &normal) : normal(normal) {
        buildOrthonormalBasis(normal, tangent, bitangent);
    }

    /// @brief Converts a vector from world-space coordinates to local shading coordinates. 
    Vector toLocal(const Vector &world) const {
        return { world.dot(tangent), world.dot(bitangent), world.dot(normal) };
    }
    /// @brief Converts a vector from local shading coordinates to world-space coordinates.
    Vector toWorld(const Vector &local) const {
        return local.x() * tangent + local.y() * bitangent + local.z() * normal;
    }
    
    /// @brief Tests whether two vectors lie within the same hemisphere (i.e., the angle they form is below 180°). 
    static bool sameHemisphere(const Vector &wi, const Vector &wo) {
        return cosTheta(wi) * cosTheta(wo) > 0;
    }

    /// @brief Computes cos(theta) for a vector in shading coordinates. 
    static float cosTheta(const Vector &w) { return w.z(); }
    /// @brief Computes cos^2(theta) for a vector in shading coordinates. 
    static float cosTheta2(const Vector &w) { return sqr(w.z()); }
    /// @brief Computes |cos(theta)| for a vector in shading coordinates. 
    static float absCosTheta(const Vector &w) {  return abs(w.z()); }
    
    /// @brief Computes sin(theta) for a vector in shading coordinates. 
    /// @brief Computes sin^2(theta) for a vector in shading coordinates. 
    static float sinTheta(const Vector &w) { return safe_sqrt(1 - cosTheta2(w)); }
    static float sinTheta2(const Vector &w) { return 1 - cosTheta2(w); }
    
    /// @brief Computes cos(phi)*sin(theta) for a vector in shading coordinates. 
    static float cosPhiSinTheta(const Vector &w) { return w.x(); }
    /// @brief Computes sin(phi)*sin(theta) for a vector in shading coordinates. 
    static float sinPhiSinTheta(const Vector &w) { return w.y(); }

    /// @brief Computes tan(theta) for a vector in shading coordinates. 
    static float tanTheta(const Vector &w) {
        const float cos = cosTheta(w);
        return safe_sqrt(1 - sqr(cos)) / cos;
    }
    
    /// @brief Computes tan^2(theta) for a vector in shading coordinates. 
    static float tanTheta2(const Vector &w) {
        const float cos2 = cosTheta2(w);
        return (1 - cos2) / cos2;
    }
};

/// @brief Barycentric interpolation ([0,0] returns a, [1,0] returns b, and [0,1] returns c).
template<typename T>
static T interpolateBarycentric(const Vector2 &bary, const T &a, const T &b, const T &c) {
    return a * (1 - bary.x() - bary.y()) + b * bary.x() + c * bary.y();
}

/// @brief Barycentric interpolation ([0,0] returns a, [1,0] returns b, and [0,1] returns c).
template<>
Point interpolateBarycentric(const Vector2 &bary, const Point &a, const Point &b, const Point &c) {
    return Point(interpolateBarycentric(bary, Vector(a), Vector(b), Vector(c)));
}

/// @brief A vertex of a triangle mesh.
struct Vertex {
    /// @brief The position of the vertex in object space.
    Point position;
    /// @brief The texture coordinates associated with the vertex.
    Vector2 texcoords;
    /// @brief The normal vector, which will be barycentrically interpolated when smooth normals are used.
    Vector normal;

    /// @brief Barycentric interpolation of vertices ([0,0] returns a, [1,0] returns b, and [0,1] returns c).
    static Vertex interpolate(const Vector2 &bary, const Vertex &a, const Vertex &b, const Vertex &c) {
        return {
            .position  = interpolateBarycentric(bary, a.position , b.position , c.position ),
            .texcoords = interpolateBarycentric(bary, a.texcoords, b.texcoords, c.texcoords),
            .normal    = interpolateBarycentric(bary, a.normal   , b.normal   , c.normal   ),
        };
    }
};

/// @brief A point on a surface along with context about the orientation of the surface.
struct SurfaceEvent {
    /// @brief The position of the surface point.
    Point position;
    /// @brief The texture coordinates of the surface for the given position.
    Point2 uv;
    /// @brief The shading frame of the surface at the given position.
    Frame frame;
    /// @brief The probability of sampling the point when doing area sampling, in area units.
    float pdf;
    /// @brief The instance object associated with the surface.
    const Instance *instance = nullptr;
};

/// @brief Describes an intersection of a ray with a surface.
struct Intersection : public SurfaceEvent {
    /// @brief The direction of the ray that hit the surface, pointing away from the surface.
    Vector wo;
    /// @brief The intersection distance, which can also be used to specify a maximum distance when querying intersections.
    float t;

    Texture* alphaMask;

    float opacity;

    /// @brief Statistics recorded while traversing acceleration structures.
    struct {
        /// @brief The number of BVH nodes that have been tested for intersection.
        int bvhCounter = 0;
        /// @brief The number of shapes that have been tested for intersection.
        int primCounter = 0;
    } stats;

    Intersection(const Vector &wo = Vector(), float t = Infinity)
    : wo(wo), t(t) {}

    Intersection(const Intersection &other) = default;
    Intersection &operator=(const Intersection &other) = default;

    /// @brief Reports whether an object has been hit.
    operator bool() const {
        return instance != nullptr;
    }

    /// @brief Evaluates the emission of the underlying instance.
    Color evaluateEmission() const;
    /// @brief Samples the Bsdf of the underlying surface.
    BsdfSample sampleBsdf(Sampler &rng) const;
    /// @brief Evaluates the Bsdf of the underlying surface.
    BsdfEval evaluateBsdf(const Vector &wi) const;
};

/// @brief Print a given point to an output stream.
template<typename Type, int Dimension>
static std::ostream &operator<<(std::ostream &os, const TPoint<Type, Dimension> &point) {
    os << "Point[";
    for (int i = 0; i < Dimension; i++) {
        if (i) os << ", ";
        os << point[i];
    }
    os << "]";
    return os;
}

/// @brief Print a given vector to an output stream.
template<typename Type, int Dimension>
static std::ostream &operator<<(std::ostream &os, const TVector<Type, Dimension> &vector) {
    os << "Vector[";
    for (int i = 0; i < Dimension; i++) {
        if (i) os << ", ";
        os << vector[i];
    }
    os << "]";
    return os;
}

/// @brief Print a given matrix to an output stream.
template<typename Type, int Rows, int Columns>
static std::ostream &operator<<(std::ostream &os, const TMatrix<Type, Rows, Columns> &matrix) {
    os << "Matrix[" << std::endl;
    for (int row = 0; row < Rows; row++) {
        os << "  ";
        for (int column = 0; column < Columns; column++) {
            os << matrix(row, column);
            os << ", ";
        }
        os << std::endl;
    }
    os << "]";
    return os;
}

}

namespace std {

static bool isfinite(const lightwave::Color &c);

/// @brief Tests whether all components of a given point are finite.
template<typename F, int D>
bool isfinite(const lightwave::TPoint<F, D> &p) {
    for (int dim = 0; dim < D; dim++)
        if (!std::isfinite(p[dim]))
            return false;
    return true;
}

}

// MARK: - assertions

namespace lightwave {

//#ifdef LW_DEBUG
// lightwave with assertions enabled is not much slower, so we also keep them enabled in Release builds
#define ENABLE_ASSERTIONS
//#endif

/// @brief Asserts that a given vector/color/value is finite, and aborts if that is note the case.
#define assert_finite(x, ctx) if (!check_finite(x, __FILE__, __LINE__)) { ctx; abort(); }
template <typename T>
inline bool check_finite(const T &n, const char *file, size_t line) {
#ifdef ENABLE_ASSERTIONS
    if (!std::isfinite(n)) {
        logger(EError, "non-finite value in %s:%d (%s)", file, line, n);
        return false;
    }
#endif
    return true;
}

/// @brief Asserts that a given vector is normalized, and aborts if that is note the case.
#define assert_normalized(x, ctx) if (!check_normalized(x, __FILE__, __LINE__)) { ctx; abort(); }
template <typename T>
inline bool check_normalized(const T &n, const char *file, size_t line) {
#ifdef ENABLE_ASSERTIONS
    float lenSqr = n.lengthSquared();
    if (!(abs(lenSqr - 1) < 0.001f)) {
        logger(EError, "vector not normalized in %s:%d (%s, lengthSquared: %f)", file, line, n, lenSqr);
        return false;
    }
#endif
    return true;
}

}

static constexpr std::array<int, 1000> Primes = {2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61, 67, 71, 73, 79, 83, 89, 97, 101, 103, 107, 109, 113, 127, 131, 137, 139, 149, 151, 157, 163, 167, 173, 179, 181, 191, 193, 197, 199, 211, 223, 227, 229, 233, 239, 241, 251, 257, 263, 269, 271, 277, 281, 283, 293, 307, 311, 313, 317, 331, 337, 347, 349, 353, 359, 367, 373, 379, 383, 389, 397, 401, 409, 419, 421, 431, 433, 439, 443, 449, 457, 461, 463, 467, 479, 487, 491, 499, 503, 509, 521, 523, 541, 547, 557, 563, 569, 571, 577, 587, 593, 599, 601, 607, 613, 617, 619, 631, 641, 643, 647, 653, 659, 661, 673, 677, 683, 691, 701, 709, 719, 727, 733, 739, 743, 751, 757, 761, 769, 773, 787, 797, 809, 811, 821, 823, 827, 829, 839, 853, 857, 859, 863, 877, 881, 883, 887, 907, 911, 919, 929, 937, 941, 947, 953, 967, 971, 977, 983, 991, 997, 1009, 1013, 1019, 1021, 1031, 1033, 1039, 1049, 1051, 1061, 1063, 1069, 1087, 1091, 1093, 1097, 1103, 1109, 1117, 1123, 1129, 1151, 1153, 1163, 1171, 1181, 1187, 1193, 1201, 1213, 1217, 1223, 1229, 1231, 1237, 1249, 1259, 1277, 1279, 1283, 1289, 1291, 1297, 1301, 1303, 1307, 1319, 1321, 1327, 1361, 1367, 1373, 1381, 1399, 1409, 1423, 1427, 1429, 1433, 1439, 1447, 1451, 1453, 1459, 1471, 1481, 1483, 1487, 1489, 1493, 1499, 1511, 1523, 1531, 1543, 1549, 1553, 1559, 1567, 1571, 1579, 1583, 1597, 1601, 1607, 1609, 1613, 1619, 1621, 1627, 1637, 1657, 1663, 1667, 1669, 1693, 1697, 1699, 1709, 1721, 1723, 1733, 1741, 1747, 1753, 1759, 1777, 1783, 1787, 1789, 1801, 1811, 1823, 1831, 1847, 1861, 1867, 1871, 1873, 1877, 1879, 1889, 1901, 1907, 1913, 1931, 1933, 1949, 1951, 1973, 1979, 1987, 1993, 1997, 1999, 2003, 2011, 2017, 2027, 2029, 2039, 2053, 2063, 2069, 2081, 2083, 2087, 2089, 2099, 2111, 2113, 2129, 2131, 2137, 2141, 2143, 2153, 2161, 2179, 2203, 2207, 2213, 2221, 2237, 2239, 2243, 2251, 2267, 2269, 2273, 2281, 2287, 2293, 2297, 2309, 2311, 2333, 2339, 2341, 2347, 2351, 2357, 2371, 2377, 2381, 2383, 2389, 2393, 2399, 2411, 2417, 2423, 2437, 2441, 2447, 2459, 2467, 2473, 2477, 2503, 2521, 2531, 2539, 2543, 2549, 2551, 2557, 2579, 2591, 2593, 2609, 2617, 2621, 2633, 2647, 2657, 2659, 2663, 2671, 2677, 2683, 2687, 2689, 2693, 2699, 2707, 2711, 2713, 2719, 2729, 2731, 2741, 2749, 2753, 2767, 2777, 2789, 2791, 2797, 2801, 2803, 2819, 2833, 2837, 2843, 2851, 2857, 2861, 2879, 2887, 2897, 2903, 2909, 2917, 2927, 2939, 2953, 2957, 2963, 2969, 2971, 2999, 3001, 3011, 3019, 3023, 3037, 3041, 3049, 3061, 3067, 3079, 3083, 3089, 3109, 3119, 3121, 3137, 3163, 3167, 3169, 3181, 3187, 3191, 3203, 3209, 3217, 3221, 3229, 3251, 3253, 3257, 3259, 3271, 3299, 3301, 3307, 3313, 3319, 3323, 3329, 3331, 3343, 3347, 3359, 3361, 3371, 3373, 3389, 3391, 3407, 3413, 3433, 3449, 3457, 3461, 3463, 3467, 3469, 3491, 3499, 3511, 3517, 3527, 3529, 3533, 3539, 3541, 3547, 3557, 3559, 3571, 3581, 3583, 3593, 3607, 3613, 3617, 3623, 3631, 3637, 3643, 3659, 3671, 3673, 3677, 3691, 3697, 3701, 3709, 3719, 3727, 3733, 3739, 3761, 3767, 3769, 3779, 3793, 3797, 3803, 3821, 3823, 3833, 3847, 3851, 3853, 3863, 3877, 3881, 3889, 3907, 3911, 3917, 3919, 3923, 3929, 3931, 3943, 3947, 3967, 3989, 4001, 4003, 4007, 4013, 4019, 4021, 4027, 4049, 4051, 4057, 4073, 4079, 4091, 4093, 4099, 4111, 4127, 4129, 4133, 4139, 4153, 4157, 4159, 4177, 4201, 4211, 4217, 4219, 4229, 4231, 4241, 4243, 4253, 4259, 4261, 4271, 4273, 4283, 4289, 4297, 4327, 4337, 4339, 4349, 4357, 4363, 4373, 4391, 4397, 4409, 4421, 4423, 4441, 4447, 4451, 4457, 4463, 4481, 4483, 4493, 4507, 4513, 4517, 4519, 4523, 4547, 4549, 4561, 4567, 4583, 4591, 4597, 4603, 4621, 4637, 4639, 4643, 4649, 4651, 4657, 4663, 4673, 4679, 4691, 4703, 4721, 4723, 4729, 4733, 4751, 4759, 4783, 4787, 4789, 4793, 4799, 4801, 4813, 4817, 4831, 4861, 4871, 4877, 4889, 4903, 4909, 4919, 4931, 4933, 4937, 4943, 4951, 4957, 4967, 4969, 4973, 4987, 4993, 4999, 5003, 5009, 5011, 5021, 5023, 5039, 5051, 5059, 5077, 5081, 5087, 5099, 5101, 5107, 5113, 5119, 5147, 5153, 5167, 5171, 5179, 5189, 5197, 5209, 5227, 5231, 5233, 5237, 5261, 5273, 5279, 5281, 5297, 5303, 5309, 5323, 5333, 5347, 5351, 5381, 5387, 5393, 5399, 5407, 5413, 5417, 5419, 5431, 5437, 5441, 5443, 5449, 5471, 5477, 5479, 5483, 5501, 5503, 5507, 5519, 5521, 5527, 5531, 5557, 5563, 5569, 5573, 5581, 5591, 5623, 5639, 5641, 5647, 5651, 5653, 5657, 5659, 5669, 5683, 5689, 5693, 5701, 5711, 5717, 5737, 5741, 5743, 5749, 5779, 5783, 5791, 5801, 5807, 5813, 5821, 5827, 5839, 5843, 5849, 5851, 5857, 5861, 5867, 5869, 5879, 5881, 5897, 5903, 5923, 5927, 5939, 5953, 5981, 5987, 6007, 6011, 6029, 6037, 6043, 6047, 6053, 6067, 6073, 6079, 6089, 6091, 6101, 6113, 6121, 6131, 6133, 6143, 6151, 6163, 6173, 6197, 6199, 6203, 6211, 6217, 6221, 6229, 6247, 6257, 6263, 6269, 6271, 6277, 6287, 6299, 6301, 6311, 6317, 6323, 6329, 6337, 6343, 6353, 6359, 6361, 6367, 6373, 6379, 6389, 6397, 6421, 6427, 6449, 6451, 6469, 6473, 6481, 6491, 6521, 6529, 6547, 6551, 6553, 6563, 6569, 6571, 6577, 6581, 6599, 6607, 6619, 6637, 6653, 6659, 6661, 6673, 6679, 6689, 6691, 6701, 6703, 6709, 6719, 6733, 6737, 6761, 6763, 6779, 6781, 6791, 6793, 6803, 6823, 6827, 6829, 6833, 6841, 6857, 6863, 6869, 6871, 6883, 6899, 6907, 6911, 6917, 6947, 6949, 6959, 6961, 6967, 6971, 6977, 6983, 6991, 6997, 7001, 7013, 7019, 7027, 7039, 7043, 7057, 7069, 7079, 7103, 7109, 7121, 7127, 7129, 7151, 7159, 7177, 7187, 7193, 7207, 7211, 7213, 7219, 7229, 7237, 7243, 7247, 7253, 7283, 7297, 7307, 7309, 7321, 7331, 7333, 7349, 7351, 7369, 7393, 7411, 7417, 7433, 7451, 7457, 7459, 7477, 7481, 7487, 7489, 7499, 7507, 7517, 7523, 7529, 7537, 7541, 7547, 7549, 7559, 7561, 7573, 7577, 7583, 7589, 7591, 7603, 7607, 7621, 7639, 7643, 7649, 7669, 7673, 7681, 7687, 7691, 7699, 7703, 7717, 7723, 7727, 7741, 7753, 7757, 7759, 7789, 7793, 7817, 7823, 7829, 7841, 7853, 7867, 7873, 7877, 7879, 7883, 7901, 7907, 7919};
static constexpr std::array<int, 1000> Prime_Sums = {0, 2, 5, 10, 17, 28, 41, 58, 77, 100, 129, 160, 197, 238, 281, 328, 381, 440, 501, 568, 639, 712, 791, 874, 963, 1060, 1161, 1264, 1371, 1480, 1593, 1720, 1851, 1988, 2127, 2276, 2427, 2584, 2747, 2914, 3087, 3266, 3447, 3638, 3831, 4028, 4227, 4438, 4661, 4888, 5117, 5350, 5589, 5830, 6081, 6338, 6601, 6870, 7141, 7418, 7699, 7982, 8275, 8582, 8893, 9206, 9523, 9854, 10191, 10538, 10887, 11240, 11599, 11966, 12339, 12718, 13101, 13490, 13887, 14288, 14697, 15116, 15537, 15968, 16401, 16840, 17283, 17732, 18189, 18650, 19113, 19580, 20059, 20546, 21037, 21536, 22039, 22548, 23069, 23592, 24133, 24680, 25237, 25800, 26369, 26940, 27517, 28104, 28697, 29296, 29897, 30504, 31117, 31734, 32353, 32984, 33625, 34268, 34915, 35568, 36227, 36888, 37561, 38238, 38921, 39612, 40313, 41022, 41741, 42468, 43201, 43940, 44683, 45434, 46191, 46952, 47721, 48494, 49281, 50078, 50887, 51698, 52519, 53342, 54169, 54998, 55837, 56690, 57547, 58406, 59269, 60146, 61027, 61910, 62797, 63704, 64615, 65534, 66463, 67400, 68341, 69288, 70241, 71208, 72179, 73156, 74139, 75130, 76127, 77136, 78149, 79168, 80189, 81220, 82253, 83292, 84341, 85392, 86453, 87516, 88585, 89672, 90763, 91856, 92953, 94056, 95165, 96282, 97405, 98534, 99685, 100838, 102001, 103172, 104353, 105540, 106733, 107934, 109147, 110364, 111587, 112816, 114047, 115284, 116533, 117792, 119069, 120348, 121631, 122920, 124211, 125508, 126809, 128112, 129419, 130738, 132059, 133386, 134747, 136114, 137487, 138868, 140267, 141676, 143099, 144526, 145955, 147388, 148827, 150274, 151725, 153178, 154637, 156108, 157589, 159072, 160559, 162048, 163541, 165040, 166551, 168074, 169605, 171148, 172697, 174250, 175809, 177376, 178947, 180526, 182109, 183706, 185307, 186914, 188523, 190136, 191755, 193376, 195003, 196640, 198297, 199960, 201627, 203296, 204989, 206686, 208385, 210094, 211815, 213538, 215271, 217012, 218759, 220512, 222271, 224048, 225831, 227618, 229407, 231208, 233019, 234842, 236673, 238520, 240381, 242248, 244119, 245992, 247869, 249748, 251637, 253538, 255445, 257358, 259289, 261222, 263171, 265122, 267095, 269074, 271061, 273054, 275051, 277050, 279053, 281064, 283081, 285108, 287137, 289176, 291229, 293292, 295361, 297442, 299525, 301612, 303701, 305800, 307911, 310024, 312153, 314284, 316421, 318562, 320705, 322858, 325019, 327198, 329401, 331608, 333821, 336042, 338279, 340518, 342761, 345012, 347279, 349548, 351821, 354102, 356389, 358682, 360979, 363288, 365599, 367932, 370271, 372612, 374959, 377310, 379667, 382038, 384415, 386796, 389179, 391568, 393961, 396360, 398771, 401188, 403611, 406048, 408489, 410936, 413395, 415862, 418335, 420812, 423315, 425836, 428367, 430906, 433449, 435998, 438549, 441106, 443685, 446276, 448869, 451478, 454095, 456716, 459349, 461996, 464653, 467312, 469975, 472646, 475323, 478006, 480693, 483382, 486075, 488774, 491481, 494192, 496905, 499624, 502353, 505084, 507825, 510574, 513327, 516094, 518871, 521660, 524451, 527248, 530049, 532852, 535671, 538504, 541341, 544184, 547035, 549892, 552753, 555632, 558519, 561416, 564319, 567228, 570145, 573072, 576011, 578964, 581921, 584884, 587853, 590824, 593823, 596824, 599835, 602854, 605877, 608914, 611955, 615004, 618065, 621132, 624211, 627294, 630383, 633492, 636611, 639732, 642869, 646032, 649199, 652368, 655549, 658736, 661927, 665130, 668339, 671556, 674777, 678006, 681257, 684510, 687767, 691026, 694297, 697596, 700897, 704204, 707517, 710836, 714159, 717488, 720819, 724162, 727509, 730868, 734229, 737600, 740973, 744362, 747753, 751160, 754573, 758006, 761455, 764912, 768373, 771836, 775303, 778772, 782263, 785762, 789273, 792790, 796317, 799846, 803379, 806918, 810459, 814006, 817563, 821122, 824693, 828274, 831857, 835450, 839057, 842670, 846287, 849910, 853541, 857178, 860821, 864480, 868151, 871824, 875501, 879192, 882889, 886590, 890299, 894018, 897745, 901478, 905217, 908978, 912745, 916514, 920293, 924086, 927883, 931686, 935507, 939330, 943163, 947010, 950861, 954714, 958577, 962454, 966335, 970224, 974131, 978042, 981959, 985878, 989801, 993730, 997661, 1001604, 1005551, 1009518, 1013507, 1017508, 1021511, 1025518, 1029531, 1033550, 1037571, 1041598, 1045647, 1049698, 1053755, 1057828, 1061907, 1065998, 1070091, 1074190, 1078301, 1082428, 1086557, 1090690, 1094829, 1098982, 1103139, 1107298, 1111475, 1115676, 1119887, 1124104, 1128323, 1132552, 1136783, 1141024, 1145267, 1149520, 1153779, 1158040, 1162311, 1166584, 1170867, 1175156, 1179453, 1183780, 1188117, 1192456, 1196805, 1201162, 1205525, 1209898, 1214289, 1218686, 1223095, 1227516, 1231939, 1236380, 1240827, 1245278, 1249735, 1254198, 1258679, 1263162, 1267655, 1272162, 1276675, 1281192, 1285711, 1290234, 1294781, 1299330, 1303891, 1308458, 1313041, 1317632, 1322229, 1326832, 1331453, 1336090, 1340729, 1345372, 1350021, 1354672, 1359329, 1363992, 1368665, 1373344, 1378035, 1382738, 1387459, 1392182, 1396911, 1401644, 1406395, 1411154, 1415937, 1420724, 1425513, 1430306, 1435105, 1439906, 1444719, 1449536, 1454367, 1459228, 1464099, 1468976, 1473865, 1478768, 1483677, 1488596, 1493527, 1498460, 1503397, 1508340, 1513291, 1518248, 1523215, 1528184, 1533157, 1538144, 1543137, 1548136, 1553139, 1558148, 1563159, 1568180, 1573203, 1578242, 1583293, 1588352, 1593429, 1598510, 1603597, 1608696, 1613797, 1618904, 1624017, 1629136, 1634283, 1639436, 1644603, 1649774, 1654953, 1660142, 1665339, 1670548, 1675775, 1681006, 1686239, 1691476, 1696737, 1702010, 1707289, 1712570, 1717867, 1723170, 1728479, 1733802, 1739135, 1744482, 1749833, 1755214, 1760601, 1765994, 1771393, 1776800, 1782213, 1787630, 1793049, 1798480, 1803917, 1809358, 1814801, 1820250, 1825721, 1831198, 1836677, 1842160, 1847661, 1853164, 1858671, 1864190, 1869711, 1875238, 1880769, 1886326, 1891889, 1897458, 1903031, 1908612, 1914203, 1919826, 1925465, 1931106, 1936753, 1942404, 1948057, 1953714, 1959373, 1965042, 1970725, 1976414, 1982107, 1987808, 1993519, 1999236, 2004973, 2010714, 2016457, 2022206, 2027985, 2033768, 2039559, 2045360, 2051167, 2056980, 2062801, 2068628, 2074467, 2080310, 2086159, 2092010, 2097867, 2103728, 2109595, 2115464, 2121343, 2127224, 2133121, 2139024, 2144947, 2150874, 2156813, 2162766, 2168747, 2174734, 2180741, 2186752, 2192781, 2198818, 2204861, 2210908, 2216961, 2223028, 2229101, 2235180, 2241269, 2247360, 2253461, 2259574, 2265695, 2271826, 2277959, 2284102, 2290253, 2296416, 2302589, 2308786, 2314985, 2321188, 2327399, 2333616, 2339837, 2346066, 2352313, 2358570, 2364833, 2371102, 2377373, 2383650, 2389937, 2396236, 2402537, 2408848, 2415165, 2421488, 2427817, 2434154, 2440497, 2446850, 2453209, 2459570, 2465937, 2472310, 2478689, 2485078, 2491475, 2497896, 2504323, 2510772, 2517223, 2523692, 2530165, 2536646, 2543137, 2549658, 2556187, 2562734, 2569285, 2575838, 2582401, 2588970, 2595541, 2602118, 2608699, 2615298, 2621905, 2628524, 2635161, 2641814, 2648473, 2655134, 2661807, 2668486, 2675175, 2681866, 2688567, 2695270, 2701979, 2708698, 2715431, 2722168, 2728929, 2735692, 2742471, 2749252, 2756043, 2762836, 2769639, 2776462, 2783289, 2790118, 2796951, 2803792, 2810649, 2817512, 2824381, 2831252, 2838135, 2845034, 2851941, 2858852, 2865769, 2872716, 2879665, 2886624, 2893585, 2900552, 2907523, 2914500, 2921483, 2928474, 2935471, 2942472, 2949485, 2956504, 2963531, 2970570, 2977613, 2984670, 2991739, 2998818, 3005921, 3013030, 3020151, 3027278, 3034407, 3041558, 3048717, 3055894, 3063081, 3070274, 3077481, 3084692, 3091905, 3099124, 3106353, 3113590, 3120833, 3128080, 3135333, 3142616, 3149913, 3157220, 3164529, 3171850, 3179181, 3186514, 3193863, 3201214, 3208583, 3215976, 3223387, 3230804, 3238237, 3245688, 3253145, 3260604, 3268081, 3275562, 3283049, 3290538, 3298037, 3305544, 3313061, 3320584, 3328113, 3335650, 3343191, 3350738, 3358287, 3365846, 3373407, 3380980, 3388557, 3396140, 3403729, 3411320, 3418923, 3426530, 3434151, 3441790, 3449433, 3457082, 3464751, 3472424, 3480105, 3487792, 3495483, 3503182, 3510885, 3518602, 3526325, 3534052, 3541793, 3549546, 3557303, 3565062, 3572851, 3580644, 3588461, 3596284, 3604113, 3611954, 3619807, 3627674, 3635547, 3643424, 3651303, 3659186, 3667087, 3674994};
