//! Implementation of [Simulation of Simplicity](https://arxiv.org/pdf/math/9410209.pdf)
//!
//! Simulation of simplicity is a technique for ignoring
//! degeneracies when calculating geometric predicates,
//! such as the orientation of one point with respect to a list of points.
//! Each point **p**\_ *i* is perturbed by some polynomial
//! in ε, a sufficiently small positive number.
//! Specifically, coordinate *p\_(i,j)* is perturbed by ε^(2^(*d*\**i* - *j*)),
//! where *d* is more than the number of dimensions.
//!
//! # Predicates
//!
//! ## Orientation
//!
//! The orientation of 2 points **p**\_0, **p**\_1 in 1-dimensional space is 
//! positive if **p**\_0 is to the right of **p**\_1 and negative otherwise.
//! We don't consider the case where **p**\_0 = **p**\_1 because of the perturbations.
//!
//! The orientation of *n* points **p**\_0, ..., **p**\_(n - 1) in (n - 1)-dimensional space is
//! the same as the orientation of **p**\_1, ..., **p**\_(n - 1) when looked at
//! from **p**_0. In particular, the orientation of 3 points in 2-dimensional space
//! is positive iff they form a left turn.
//!
//! Orientation predicates for 1, 2, 3, and 4 dimensions are implemented.
//! They return whether the orientation is positive.
//!
//! # Usage
//!
//! ```rust
//! use simulation_of_simplicity::{nalgebra, orientation_2d};
//! use nalgebra::Vector2;
//!
//! let points = vec![
//!     Vector2::new(0.0, 0.0),
//!     Vector2::new(1.0, 0.0),
//!     Vector2::new(1.0, 1.0),
//!     Vector2::new(0.0, 1.0),
//!     Vector2::new(2.0, 0.0),
//! ];
//!
//! // Positive orientation
//! let result = orientation_2d(&points, |l, i| l[i], [0, 1, 2]);
//! assert!(result);
//!
//! // Negative orientation
//! let result = orientation_2d(&points, |l, i| l[i], [0, 3, 2]);
//! assert!(!result);
//!
//! // Degenerate orientation, tie broken by perturbance
//! let result = orientation_2d(&points, |l, i| l[i], [0, 1, 4]);
//! assert!(result);
//! let result = orientation_2d(&points, |l, i| l[i], [4, 1, 0]);
//! assert!(!result);
//! ```
//!
//! Because the predicates take an indexing function, this can be
//! used for arbitrary lists without having to implement `Index` for them:
//!
//! ```rust
//! # use simulation_of_simplicity::{nalgebra, orientation_2d};
//! # use nalgebra::Vector2;
//! let points = vec![
//!     (Vector2::new(0.0, 0.0), 0.8),
//!     (Vector2::new(1.0, 0.0), 0.4),
//!     (Vector2::new(2.0, 0.0), 0.6),
//! ];
//!
//! let result = orientation_2d(&points, |l, i| l[i].0, [0, 1, 2]);
//! ```

pub extern crate nalgebra;

use nalgebra::{Vector1, Vector2, Vector3, Vector4};
type Vec1 = Vector1<f64>;
type Vec2 = Vector2<f64>;
type Vec3 = Vector3<f64>;
type Vec4 = Vector4<f64>;

/// Returns whether the orientation of 2 points in 1-dimensional space
/// is positive after perturbing them; that is, if the 1st one is
/// to the right of the 2nd one.
///
/// Takes a list of all the points in consideration, an indexing function,
/// and an array of 2 indexes to the points to calculate the orientation of.
///
/// # Example
///
/// ```
/// # use simulation_of_simplicity::{nalgebra, orientation_1d};
/// # use nalgebra::Vector1;
/// let points = vec![0.0, 1.0, 2.0, 1.0];
/// let positive = orientation_1d(&points, |l, i| Vector1::new(l[i]), [1, 3]);
/// // points[1] gets perturbed farther to the right than points[3]
/// assert!(positive);
/// ```
pub fn orientation_1d<T>(list: &T, index_fn: impl Fn(&T, usize) -> Vec1, indexes: [usize; 2]) -> bool {
    let p0 = index_fn(list, indexes[0]);
    let p1 = index_fn(list, indexes[1]);
    p0 > p1 || (p0 == p1 && indexes[0] < indexes[1])
}

/// Returns whether the orientation of 3 points in 2-dimensional space
/// is positive after perturbing them; that is, if the 3 points
/// form a left turn when visited in order.
///
/// Takes a list of all the points in consideration, an indexing function,
/// and an array of 3 indexes to the points to calculate the orientation of.
///
/// # Example
///
/// ```
/// # use simulation_of_simplicity::{nalgebra, orientation_2d};
/// # use nalgebra::Vector2;
/// let points = vec![
///     Vector2::new(0.0, 0.0),
///     Vector2::new(1.0, 0.0),
///     Vector2::new(1.0, 1.0),
///     Vector2::new(2.0, 2.0),
/// ];
/// let positive = orientation_2d(&points, |l, i| l[i], [0, 1, 2]);
/// assert!(positive);
/// let positive = orientation_2d(&points, |l, i| l[i], [0, 3, 2]);
/// assert!(!positive);
/// ```
pub fn orientation_2d<T>(list: &T, index_fn: impl Fn(&T, usize) -> Vec2, indexes: [usize; 3]) -> bool {
    todo!()
}

pub fn orientation_3d<T>(list: &T, index_fn: impl Fn(&T, usize) -> Vec3, indexes: [usize; 4]) -> bool {
    todo!()
}

pub fn orientation_4d<T>(list: &T, index_fn: impl Fn(&T, usize) -> Vec4, indexes: [usize; 5]) -> bool {
    todo!()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn orientation_1d_positive() {
        let points = vec![0.0, 1.0];
        assert!(orientation_1d(&points, |l, i| Vector1::new(l[i]), [1, 0]))
    }

    #[test]
    fn orientation_1d_negative() {
        let points = vec![0.0, 1.0];
        assert!(!orientation_1d(&points, |l, i| Vector1::new(l[i]), [0, 1]))
    }

    #[test]
    fn orientation_1d_positive_degenerate() {
        let points = vec![0.0, 0.0];
        assert!(orientation_1d(&points, |l, i| Vector1::new(l[i]), [0, 1]))
    }

    #[test]
    fn orientation_1d_negative_degenerate() {
        let points = vec![0.0, 0.0];
        assert!(!orientation_1d(&points, |l, i| Vector1::new(l[i]), [1, 0]))
    }
}
