//! Implementation of [Simulation of Simplicity by Edelsbrunner and Mücke](https://arxiv.org/pdf/math/9410209.pdf)
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
//! Orientation predicates for 1, 2, and 3 dimensions are implemented.
//! They return whether the orientation is positive.
//!
//! ## In Hypersphere
//!
//! The in-circle of 4 points measures whether the last point is inside
//! the circle that goes through the first 3 points. Those 3 points
//! are not collinear because of the perturbations.
//!
//! The in-sphere of 5 points measures whether the last point is inside
//! the sphere that goes through the first 4 points. Those 4 points
//! are not coplanar because of the perturbations.
//!
//! # Usage
//!
//! ```rust
//! use simplicity::{nalgebra, orient_2d};
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
//! let result = orient_2d(&points, |l, i| l[i], 0, 1, 2);
//! assert!(result);
//!
//! // Negative orientation
//! let result = orient_2d(&points, |l, i| l[i], 0, 3, 2);
//! assert!(!result);
//!
//! // Degenerate orientation, tie broken by perturbance
//! let result = orient_2d(&points, |l, i| l[i], 0, 1, 4);
//! assert!(result);
//! let result = orient_2d(&points, |l, i| l[i], 4, 1, 0);
//! assert!(!result);
//! ```
//!
//! Because the predicates take an indexing function, this can be
//! used for arbitrary lists without having to implement `Index` for them:
//!
//! ```rust
//! # use simplicity::{nalgebra, orient_2d};
//! # use nalgebra::Vector2;
//! let points = vec![
//!     (Vector2::new(0.0, 0.0), 0.8),
//!     (Vector2::new(1.0, 0.0), 0.4),
//!     (Vector2::new(2.0, 0.0), 0.6),
//! ];
//!
//! let result = orient_2d(&points, |l, i| l[i].0, 0, 1, 2);
//! ```

use robust_geo as rg;
pub use nalgebra;

use nalgebra::{Vector1, Vector2, Vector3};
type Vec1 = Vector1<f64>;
type Vec2 = Vector2<f64>;
type Vec3 = Vector3<f64>;

macro_rules! sorted_fn {
    ($name:ident, $n:expr) => {
        /// Sorts an array of $n elements
        /// and returns the sorted array,
        /// along with the parity of the permutation;
        /// `false` if even and `true` if odd.
        fn $name(mut arr: [usize; $n]) -> ([usize; $n], bool) {
            let mut num_swaps = 0;

            for i in 1..$n {
                for j in (0..i).rev() {
                    if arr[j] > arr[j + 1] {
                        arr.swap(j, j + 1);
                        num_swaps += 1;
                    } else {
                        break;
                    }
                }
            }
            (arr, num_swaps % 2 != 0)
        }
    };
}

sorted_fn!(sorted_3, 3);
sorted_fn!(sorted_4, 4);
sorted_fn!(sorted_5, 5);

/// Returns whether the orientation of 2 points in 1-dimensional space
/// is positive after perturbing them; that is, if the 1st one is
/// to the right of the 2nd one.
///
/// Takes a list of all the points in consideration, an indexing function,
/// and 2 indexes to the points to calculate the orientation of.
///
/// # Example
///
/// ```
/// # use simplicity::{nalgebra, orient_1d};
/// # use nalgebra::Vector1;
/// let points = vec![0.0, 1.0, 2.0, 1.0];
/// let positive = orient_1d(&points, |l, i| Vector1::new(l[i]), 1, 3);
/// // points[1] gets perturbed farther to the right than points[3]
/// assert!(positive);
/// ```
pub fn orient_1d<T: ?Sized>(
    list: &T,
    index_fn: impl Fn(&T, usize) -> Vec1,
    i: usize,
    j: usize,
) -> bool {
    let pi = index_fn(list, i);
    let pj = index_fn(list, j);
    pi > pj || (pi == pj && i < j)
}

macro_rules! case {
    (2: $pi:ident, $pj:ident, @ m2, != $odd:expr) => {
        let val = rg::magnitude_cmp_2d($pi, $pj);
        if val != 0.0 {
            return (val > 0.0) != $odd;
        }
    };

    (2: $pi:ident, $pj:ident, @ m3, != $odd:expr) => {
        let val = rg::magnitude_cmp_3d($pi, $pj);
        if val != 0.0 {
            return (val > 0.0) != $odd;
        }
    };

    (2: $pi:ident, $pj:ident, $(@ $swiz:ident,)? != $odd:expr) => {
        if $pi$(.$swiz)? != $pj$(.$swiz)? {
            return ($pi$(.$swiz)? > $pj$(.$swiz)?) != $odd;
        }
    };

    (3: $pi:ident, $pj:ident, $pk:ident, @ $swiz:ident m2, != $odd:expr) => {
        let val = rg::sign_det_x_x2y2($pi.$swiz(), $pj.$swiz(), $pk.$swiz());
        if val != 0.0 {
            return (val > 0.0) != $odd;
        }
    };

    (3: $pi:ident, $pj:ident, $pk:ident, @ $swiz:ident m3, != $odd:expr) => {
        let val = rg::sign_det_x_x2y2z2($pi.$swiz(), $pj.$swiz(), $pk.$swiz());
        if val != 0.0 {
            return (val > 0.0) != $odd;
        }
    };

    (3: $pi:ident, $pj:ident, $pk:ident, $(@ $swiz:ident,)? != $odd:expr) => {
        let val = rg::orient_2d($pi$(.$swiz())?, $pj$(.$swiz())?, $pk$(.$swiz())?);
        if val != 0.0 {
            return (val > 0.0) != $odd;
        }
    };

    (4: $pi:ident, $pj:ident, $pk:ident, $pl:ident, @ xy m2, != $odd:expr) => {
        let val = rg::in_circle($pi, $pj, $pk, $pl);
        if val != 0.0 {
            return (val > 0.0) != $odd;
        }
    };

    (4: $pi:ident, $pj:ident, $pk:ident, $pl:ident, @ $swiz:ident m3, != $odd:expr) => {
        let val = rg::sign_det_x_y_x2y2z2($pi.$swiz(), $pj.$swiz(), $pk.$swiz(), $pl.$swiz());
        if val != 0.0 {
            return (val > 0.0) != $odd;
        }
    };

    (4: $pi:ident, $pj:ident, $pk:ident, $pl:ident, $(@ $swiz:ident,)? != $odd:expr) => {
        let val = rg::orient_3d($pi$(.$swiz())?, $pj$(.$swiz())?, $pk$(.$swiz())?, $pl$(.$swiz())?);
        if val != 0.0 {
            return (val > 0.0) != $odd;
        }
    };

    (5: $pi:ident, $pj:ident, $pk:ident, $pl:ident, $pm:ident, @ xyz m3, != $odd:expr) => {
        let val = rg::in_sphere($pi, $pj, $pk, $pl, $pm);
        if val != 0.0 {
            return (val > 0.0) != $odd;
        }
    };

}

/// Returns whether the orientation of 3 points in 2-dimensional space
/// is positive after perturbing them; that is, if the 3 points
/// form a left turn when visited in order.
///
/// Takes a list of all the points in consideration, an indexing function,
/// and 3 indexes to the points to calculate the orientation of.
///
/// # Example
///
/// ```
/// # use simplicity::{nalgebra, orient_2d};
/// # use nalgebra::Vector2;
/// let points = vec![
///     Vector2::new(0.0, 0.0),
///     Vector2::new(1.0, 0.0),
///     Vector2::new(1.0, 1.0),
///     Vector2::new(2.0, 2.0),
/// ];
/// let positive = orient_2d(&points, |l, i| l[i], 0, 1, 2);
/// assert!(positive);
/// let positive = orient_2d(&points, |l, i| l[i], 0, 3, 2);
/// assert!(!positive);
/// ```
pub fn orient_2d<T: ?Sized>(
    list: &T,
    index_fn: impl Fn(&T, usize) -> Vec2,
    i: usize,
    j: usize,
    k: usize,
) -> bool {
    let ([i, j, k], odd) = sorted_3([i, j, k]);
    let pi = index_fn(list, i);
    let pj = index_fn(list, j);
    let pk = index_fn(list, k);

    case!(3: pi, pj, pk, != odd);
    case!(2: pk, pj, @ x, != odd);
    case!(2: pj, pk, @ y, != odd);
    case!(2: pi, pk, @ x, != odd);
    !odd
}

/// Returns whether the orientation of 4 points in 3-dimensional space
/// is positive after perturbing them; that is, if the last 3 points
/// form a left turn when visited in order, looking from the first point.
///
/// Takes a list of all the points in consideration, an indexing function,
/// and 4 indexes to the points to calculate the orientation of.
///
/// # Example
///
/// ```
/// # use simplicity::{nalgebra, orient_3d};
/// # use nalgebra::Vector3;
/// let points = vec![
///     Vector3::new(0.0, 0.0, 0.0),
///     Vector3::new(1.0, 0.0, 0.0),
///     Vector3::new(1.0, 1.0, 1.0),
///     Vector3::new(2.0, -2.0, 0.0),
///     Vector3::new(2.0, 3.0, 4.0),
///     Vector3::new(0.0, 0.0, 1.0),
///     Vector3::new(0.0, 1.0, 0.0),
///     Vector3::new(3.0, 4.0, 5.0),
/// ];
/// let positive = orient_3d(&points, |l, i| l[i], 0, 1, 6, 5);
/// assert!(!positive);
/// let positive = orient_3d(&points, |l, i| l[i], 7, 4, 0, 2);
/// assert!(positive);
/// ```
pub fn orient_3d<T: ?Sized>(
    list: &T,
    index_fn: impl Fn(&T, usize) -> Vec3,
    i: usize,
    j: usize,
    k: usize,
    l: usize,
) -> bool {
    let ([i, j, k, l], odd) = sorted_4([i, j, k, l]);
    let pi = index_fn(list, i);
    let pj = index_fn(list, j);
    let pk = index_fn(list, k);
    let pl = index_fn(list, l);

    case!(4: pi, pj, pk, pl, != odd);
    case!(3: pj, pk, pl, @ xy, != odd);
    case!(3: pj, pk, pl, @ zx, != odd);
    case!(3: pj, pk, pl, @ yz, != odd);
    case!(3: pi, pk, pl, @ yx, != odd);
    case!(2: pk, pl, @ x, != odd);
    case!(2: pl, pk, @ y, != odd);
    case!(3: pi, pk, pl, @ xz, != odd);
    case!(2: pk, pl, @ z, != odd);
    // case!(3: pi, pk, pl, @ zy, != odd); Impossible
    case!(3: pi, pj, pl, @ xy, != odd);
    case!(2: pl, pj, @ x, != odd);
    case!(2: pj, pl, @ y, != odd);
    case!(2: pi, pl, @ x, != odd);
    !odd
}

/// Returns whether the last point is inside the circle that goes through
/// the first 3 points after perturbing them.
///
/// Takes a list of all the points in consideration, an indexing function,
/// and 4 indexes to the points to calculate the in-circle of.
///
/// # Example
///
/// ```
/// # use simplicity::{nalgebra, in_circle};
/// # use nalgebra::Vector2;
/// let points = vec![
///     Vector2::new(0.0, 2.0),
///     Vector2::new(1.0, 1.0),
///     Vector2::new(2.0, 1.0),
///     Vector2::new(0.0, 0.0),
///     Vector2::new(2.0, 3.0),
/// ];
/// let inside = in_circle(&points, |l, i| l[i], 0, 2, 3, 1);
/// assert!(inside);
/// let inside = in_circle(&points, |l, i| l[i], 2, 3, 1, 4);
/// assert!(!inside);
/// ```
pub fn in_circle<T: ?Sized>(
    list: &T,
    index_fn: impl Fn(&T, usize) -> Vec2 + Clone,
    i: usize,
    j: usize,
    k: usize,
    l: usize,
) -> bool {
    let flip = !orient_2d(list, index_fn.clone(), i, j, k);
    let ([i, j, k, l], odd) = sorted_4([i, j, k, l]);
    let odd = odd != flip;

    let pi = index_fn(list, i);
    let pj = index_fn(list, j);
    let pk = index_fn(list, k);
    let pl = index_fn(list, l);

    case!(4: pi, pj, pk, pl, @ xy m2, != odd);
    case!(3: pj, pk, pl, @ xy, != odd);
    case!(3: pj, pl, pk, @ xy m2, != odd);
    case!(3: pj, pk, pl, @ yx m2, != odd);
    case!(3: pi, pk, pl, @ yx, != odd);
    case!(2: pk, pl, @ x, != odd);
    case!(2: pl, pk, @ y, != odd);
    // case!(3: pi, pk, pl, @ xy m2, != odd); Impossible
    // case!(2: pk, pl, @ m2, != odd); Impossible
    // case!(3: pi, pk, pl, @ zy, != odd); Impossible
    case!(3: pi, pj, pl, @ xy, != odd);
    case!(2: pl, pj, @ x, != odd);
    case!(2: pj, pl, @ y, != odd);
    case!(2: pi, pl, @ x, != odd);
    !odd
}

/// Returns whether the last point is inside the sphere that goes through
/// the first 4 points after perturbing them.
///
/// Takes a list of all the points in consideration, an indexing function,
/// and 5 indexes to the points to calculate the in-sphere of.
///
/// # Example
///
/// ```
/// # use simplicity::{nalgebra, in_sphere};
/// # use nalgebra::Vector3;
/// let points = vec![
///     Vector3::new(0.0, 0.0, 0.0),
///     Vector3::new(4.0, 0.0, 0.0),
///     Vector3::new(0.0, 4.0, 0.0),
///     Vector3::new(0.0, 0.0, 4.0),
///     Vector3::new(1.0, 1.0, 1.0),
/// ];
/// let inside = in_sphere(&points, |l, i| l[i], 0, 2, 3, 1, 4);
/// assert!(inside);
/// let inside = in_sphere(&points, |l, i| l[i], 2, 3, 1, 4, 0);
/// assert!(!inside);
/// ```
pub fn in_sphere<T: ?Sized>(
    list: &T,
    index_fn: impl Fn(&T, usize) -> Vec3 + Clone,
    i: usize,
    j: usize,
    k: usize,
    l: usize,
    m: usize,
) -> bool {
    let flip = !orient_3d(list, index_fn.clone(), i, j, k, l);
    let ([i, j, k, l, m], odd) = sorted_5([i, j, k, l, m]);
    let odd = odd != flip;

    let pi = index_fn(list, i);
    let pj = index_fn(list, j);
    let pk = index_fn(list, k);
    let pl = index_fn(list, l);
    let pm = index_fn(list, m);

    case!(5: pi, pj, pk, pl, pm, @ xyz m3, != odd);
    case!(4: pj, pk, pm, pl, != odd);
    case!(4: pj, pk, pl, pm, @ xyz m3, != odd);
    case!(4: pj, pk, pl, pm, @ zxy m3, != odd);
    case!(4: pj, pk, pl, pm, @ yzx m3, != odd);
    case!(4: pi, pk, pl, pm, != odd);
    case!(3: pk, pl, pm, @ xy, != odd);
    case!(3: pk, pl, pm, @ zx, != odd);
    case!(3: pk, pl, pm, @ yz, != odd);
    case!(4: pi, pk, pl, pm, @ yxz m3, != odd);
    case!(3: pk, pl, pm, @ xyz m3, != odd);
    case!(3: pk, pm, pl, @ yzx m3, != odd);
    case!(4: pi, pk, pl, pm, @ xzy m3, != odd);
    case!(3: pk, pl, pm, @ zxy m3, != odd);
    case!(4: pi, pk, pl, pm, @ zyx m3, != odd);
    case!(4: pi, pj, pm, pl, != odd);
    case!(3: pj, pl, pm, @ yx, != odd);
    case!(3: pj, pl, pm, @ xz, != odd);
    case!(3: pj, pl, pm, @ zy, != odd);
    case!(3: pi, pl, pm, @ xy, != odd);
    case!(2: pm, pl, @ x, != odd);
    case!(2: pl, pm, @ y, != odd);
    case!(3: pi, pl, pm, @ zx, != odd);
    case!(2: pm, pl, @ z, != odd);
    case!(3: pi, pl, pm, @ yz, != odd);
    case!(4: pi, pj, pl, pm, @ xyz m3, != odd);
    case!(3: pj, pm, pl, @ xyz m3, != odd);
    case!(3: pj, pl, pm, @ yzx m3, != odd);
    case!(3: pi, pl, pm, @ xyz m3, != odd);
    case!(2: pl, pm, @ m3, != odd);
    case!(3: pi, pm, pl, @ yzx m3, != odd);
    case!(4: pi, pj, pl, pm, @ zxy m3, != odd);
    case!(3: pj, pm, pl, @ zxy m3, != odd);
    case!(3: pi, pl, pm, @ zxy m3, != odd);
    case!(4: pi, pj, pl, pm, @ yzx m3, != odd);
    case!(4: pi, pj, pk, pm, != odd);
    case!(3: pj, pk, pm, @ xy, != odd);
    case!(3: pj, pk, pm, @ zx, != odd);
    case!(3: pj, pk, pm, @ yz, != odd);
    case!(3: pi, pk, pm, @ yx, != odd);
    case!(2: pk, pm, @ x, != odd);
    case!(2: pm, pk, @ y, != odd);
    case!(3: pi, pk, pm, @ xz, != odd);
    case!(2: pk, pm, @ z, != odd);
    // case!(3: pi, pk, pm, @ zy, != odd); Impossible
    case!(3: pi, pj, pm, @ xy, != odd);
    case!(2: pm, pj, @ x, != odd);
    case!(2: pj, pm, @ y, != odd);
    case!(2: pi, pm, @ x, != odd);
    !odd
}

#[cfg(test)]
mod tests {
    use super::*;
    use test_case::test_case;

    // Test-specific to determine case reached
    macro_rules! case {
        ($arr:expr => $pi:ident, $pj:ident, @ m2) => {
            let val = rg::magnitude_cmp_2d($pi, $pj);
            if val != 0.0 {
                return $arr;
            }
        };

        ($arr:expr => $pi:ident, $pj:ident, @ m3) => {
            let val = rg::magnitude_cmp_3d($pi, $pj);
            if val != 0.0 {
                return $arr;
            }
        };

        ($arr:expr => $pi:ident, $pj:ident $(, @ $swiz:ident)?) => {
            if $pi$(.$swiz)? != $pj$(.$swiz)? {
                return $arr;
            }
        };

        ($arr:expr => $pi:ident, $pj:ident, $pk:ident, @ $swiz:ident m2) => {
            let val = rg::sign_det_x_x2y2($pi.$swiz(), $pj.$swiz(), $pk.$swiz());
            if val != 0.0 {
                return $arr;
            }
        };

        ($arr:expr => $pi:ident, $pj:ident, $pk:ident, @ $swiz:ident m3) => {
            let val = rg::sign_det_x_x2y2z2($pi.$swiz(), $pj.$swiz(), $pk.$swiz());
            if val != 0.0 {
                return $arr;
            }
        };

        ($arr:expr => $pi:ident, $pj:ident, $pk:ident $(, @ $swiz:ident)?) => {
            let val = rg::orient_2d($pi$(.$swiz())?, $pj$(.$swiz())?, $pk$(.$swiz())?);
            if val != 0.0 {
                return $arr;
            }
        };

        ($arr:expr => $pi:ident, $pj:ident, $pk:ident, $pl:ident, @ xy m2) => {
            let val = rg::in_circle($pi, $pj, $pk, $pl);
            if val != 0.0 {
                return $arr;
            }
        };

        ($arr:expr => $pi:ident, $pj:ident, $pk:ident, $pl:ident, @ $swiz:ident m3) => {
            let val = rg::sign_det_x_y_x2y2z2($pi.$swiz(), $pj.$swiz(), $pk.$swiz(), $pl.$swiz());
            if val != 0.0 {
                return $arr;
            }
        };

        ($arr:expr => $pi:ident, $pj:ident, $pk:ident, $pl:ident $(, @ $swiz:ident)?) => {
            let val = rg::orient_3d($pi$(.$swiz())?, $pj$(.$swiz())?, $pk$(.$swiz())?, $pl$(.$swiz())?);
            if val != 0.0 {
                return $arr;
            }
        };

        ($arr:expr => $pi:ident, $pj:ident, $pk:ident, $pl:ident, $pm:ident, @ xyz m3) => {
            let val = rg::in_sphere($pi, $pj, $pk, $pl, $pm);
            if val != 0.0 {
                return $arr;
            }
        };
    }

    // Copied from orient_2d
    pub fn orient_2d_case<T: ?Sized>(
        list: &T,
        index_fn: impl Fn(&T, usize) -> Vec2,
        i: usize,
        j: usize,
        k: usize,
    ) -> [usize; 3] {
        let ([i, j, k], _) = sorted_3([i, j, k]);
        let pi = index_fn(list, i);
        let pj = index_fn(list, j);
        let pk = index_fn(list, k);

        case!([3, 3, 3] => pi, pj, pk);
        case!([2, 3, 3] => pk, pj, @ x);
        case!([1, 3, 3] => pj, pk, @ y);
        case!([2, 2, 3] => pi, pk, @ x);
        [1, 2, 3]
    }

    // Copied from orient_3d
    pub fn orient_3d_case<T: ?Sized>(
        list: &T,
        index_fn: impl Fn(&T, usize) -> Vec3,
        i: usize,
        j: usize,
        k: usize,
        l: usize,
    ) -> [usize; 4] {
        let ([i, j, k, l], _) = sorted_4([i, j, k, l]);
        let pi = index_fn(list, i);
        let pj = index_fn(list, j);
        let pk = index_fn(list, k);
        let pl = index_fn(list, l);

        case!([4, 4, 4, 4] => pi, pj, pk, pl);
        case!([3, 4, 4, 4] => pj, pk, pl, @ xy);
        case!([2, 4, 4, 4] => pj, pk, pl, @ zx);
        case!([1, 4, 4, 4] => pj, pk, pl, @ yz);
        case!([3, 3, 4, 4] => pi, pk, pl, @ yx);
        case!([2, 3, 4, 4] => pk, pl, @ x);
        case!([1, 3, 4, 4] => pl, pk, @ y);
        case!([2, 2, 4, 4] => pi, pk, pl, @ xz);
        case!([1, 2, 4, 4] => pk, pl, @ z);
        //case!([1, 1, 4, 4] => pi, pk, pl, @ zy); Impossible
        case!([3, 3, 3, 4] => pi, pj, pl, @ xy);
        case!([2, 3, 3, 4] => pl, pj, @ x);
        case!([1, 3, 3, 4] => pj, pl, @ y);
        case!([2, 2, 3, 4] => pi, pl, @ x);
        [1, 2, 3, 4]
    }

    pub fn in_circle_case<T: ?Sized>(
        list: &T,
        index_fn: impl Fn(&T, usize) -> Vec2 + Clone,
        i: usize,
        j: usize,
        k: usize,
        l: usize,
    ) -> [usize; 4] {
        let ([i, j, k, l], _) = sorted_4([i, j, k, l]);

        let pi = index_fn(list, i);
        let pj = index_fn(list, j);
        let pk = index_fn(list, k);
        let pl = index_fn(list, l);

        case!([4, 4, 4, 4] => pi, pj, pk, pl, @ xy m2);
        case!([3, 4, 4, 4] => pj, pk, pl, @ xy);
        case!([2, 4, 4, 4] => pj, pl, pk, @ xy m2);
        case!([1, 4, 4, 4] => pj, pk, pl, @ yx m2);
        case!([3, 3, 4, 4] => pi, pk, pl, @ yx);
        case!([2, 3, 4, 4] => pk, pl, @ x);
        case!([1, 3, 4, 4] => pl, pk, @ y);
        //case!([2, 2, 4, 4] => pi, pk, pl, @ xy m2); Impossible
        //case!([1, 2, 4, 4] => pk, pl, @ m2); Impossible
        //case!([1, 1, 4, 4] => pi, pl, pk, @ yx m2); Impossible
        case!([3, 3, 3, 4] => pi, pj, pl, @ xy);
        case!([2, 3, 3, 4] => pl, pj, @ x);
        case!([1, 3, 3, 4] => pj, pl, @ y);
        case!([2, 2, 3, 4] => pi, pl, @ x);
        [1, 2, 3, 4]
    }

    pub fn in_sphere_case<T: ?Sized>(
        list: &T,
        index_fn: impl Fn(&T, usize) -> Vec3 + Clone,
        i: usize,
        j: usize,
        k: usize,
        l: usize,
        m: usize,
    ) -> [usize; 5] {
        let ([i, j, k, l, m], _) = sorted_5([i, j, k, l, m]);

        let pi = index_fn(list, i);
        let pj = index_fn(list, j);
        let pk = index_fn(list, k);
        let pl = index_fn(list, l);
        let pm = index_fn(list, m);

        case!([5, 5, 5, 5, 5] => pi, pj, pk, pl, pm, @ xyz m3);
        case!([4, 5, 5, 5, 5] => pj, pk, pm, pl);
        case!([3, 5, 5, 5, 5] => pj, pk, pl, pm, @ xyz m3);
        case!([2, 5, 5, 5, 5] => pj, pk, pl, pm, @ zxy m3);
        case!([1, 5, 5, 5, 5] => pj, pk, pl, pm, @ yzx m3);
        case!([4, 4, 5, 5, 5] => pi, pk, pl, pm);
        case!([3, 4, 5, 5, 5] => pk, pl, pm, @ xy);
        case!([2, 4, 5, 5, 5] => pk, pl, pm, @ zx);
        case!([1, 4, 5, 5, 5] => pk, pl, pm, @ yz);
        case!([3, 3, 5, 5, 5] => pi, pk, pl, pm, @ yxz m3);
        case!([2, 3, 5, 5, 5] => pk, pl, pm, @ xyz m3);
        case!([1, 3, 5, 5, 5] => pk, pm, pl, @ yzx m3);
        case!([2, 2, 5, 5, 5] => pi, pk, pl, pm, @ xzy m3);
        case!([1, 2, 5, 5, 5] => pk, pl, pm, @ zxy m3);
        case!([1, 1, 5, 5, 5] => pi, pk, pl, pm, @ zyx m3);
        case!([4, 4, 4, 5, 5] => pi, pj, pm, pl);
        case!([3, 4, 4, 5, 5] => pj, pl, pm, @ yx);
        case!([2, 4, 4, 5, 5] => pj, pl, pm, @ xz);
        case!([1, 4, 4, 5, 5] => pj, pl, pm, @ zy);
        case!([3, 3, 4, 5, 5] => pi, pl, pm, @ xy);
        case!([2, 3, 4, 5, 5] => pm, pl, @ x);
        case!([1, 3, 4, 5, 5] => pl, pm, @ y);
        case!([2, 2, 4, 5, 5] => pi, pl, pm, @ zx);
        case!([1, 2, 4, 5, 5] => pm, pl, @ z);
        case!([1, 1, 4, 5, 5] => pi, pl, pm, @ yz);
        case!([3, 3, 3, 5, 5] => pi, pj, pl, pm, @ xyz m3);
        case!([2, 3, 3, 5, 5] => pj, pm, pl, @ xyz m3);
        case!([1, 3, 3, 5, 5] => pj, pl, pm, @ yzx m3);
        case!([2, 2, 3, 5, 5] => pi, pl, pm, @ xyz m3);
        case!([1, 2, 3, 5, 5] => pl, pm, @ m3);
        case!([1, 1, 3, 5, 5] => pi, pm, pl, @ yzx m3);
        case!([2, 2, 2, 5, 5] => pi, pj, pl, pm, @ zxy m3);
        case!([1, 2, 2, 5, 5] => pj, pm, pl, @ zxy m3);
        case!([1, 1, 2, 5, 5] => pi, pl, pm, @ zxy m3);
        case!([1, 1, 1, 5, 5] => pi, pj, pl, pm, @ yzx m3);
        case!([4, 4, 4, 4, 5] => pi, pj, pk, pm);
        case!([3, 4, 4, 4, 5] => pj, pk, pm, @ xy);
        case!([2, 4, 4, 4, 5] => pj, pk, pm, @ zx);
        case!([1, 4, 4, 4, 5] => pj, pk, pm, @ yz);
        case!([3, 3, 4, 4, 5] => pi, pk, pm, @ yx);
        case!([2, 3, 4, 4, 5] => pk, pm, @ x);
        case!([1, 3, 4, 4, 5] => pm, pk, @ y);
        case!([2, 2, 4, 4, 5] => pi, pk, pm, @ xz);
        case!([1, 2, 4, 4, 5] => pk, pm, @ z);
        // case!([1, 1, 4, 4, 5] => pi, pk, pm, @ zy); Impossible
        case!([3, 3, 3, 4, 5] => pi, pj, pm, @ xy);
        case!([2, 3, 3, 4, 5] => pm, pj, @ x);
        case!([1, 3, 3, 4, 5] => pj, pm, @ y);
        case!([2, 2, 3, 4, 5] => pi, pm, @ x);
        [1, 2, 3, 4, 5]
    }

    #[test]
    fn orient_1d_positive() {
        let points = vec![0.0, 1.0];
        assert!(orient_1d(&points, |l, i| Vector1::new(l[i]), 1, 0))
    }

    #[test]
    fn orient_1d_negative() {
        let points = vec![0.0, 1.0];
        assert!(!orient_1d(&points, |l, i| Vector1::new(l[i]), 0, 1))
    }

    #[test]
    fn orient_1d_positive_degenerate() {
        let points = vec![0.0, 0.0];
        assert!(orient_1d(&points, |l, i| Vector1::new(l[i]), 0, 1))
    }

    #[test]
    fn orient_1d_negative_degenerate() {
        let points = vec![0.0, 0.0];
        assert!(!orient_1d(&points, |l, i| Vector1::new(l[i]), 1, 0))
    }

    #[test_case([[0.0, 0.0], [1.0, 0.0], [2.0, 1.0]], [3,3,3] ; "General")]
    #[test_case([[0.0, 0.0], [1.0, 1.0], [2.0, 2.0]], [2,3,3] ; "Collinear")]
    #[test_case([[0.0, 0.0], [0.0, 2.0], [0.0, 1.0]], [1,3,3] ; "Collinear, pj.x = pk.x")]
    #[test_case([[1.0, 0.0], [0.0, 2.0], [0.0, 2.0]], [2,2,3] ; "pj = pk")]
    #[test_case([[0.0, 0.0], [0.0, 2.0], [0.0, 2.0]], [1,2,3] ; "pj = pk, pi.x = pk.x")]
    fn test_orient_2d(points: [[f64; 2]; 3], case: [usize; 3]) {
        let points = points
            .iter()
            .copied()
            .map(Vector2::from)
            .collect::<Vec<_>>();
        assert!(orient_2d(&points, |l, i| l[i], 0, 1, 2));
        assert!(!orient_2d(&points, |l, i| l[i], 0, 2, 1));
        assert!(!orient_2d(&points, |l, i| l[i], 1, 0, 2));
        assert!(orient_2d(&points, |l, i| l[i], 1, 2, 0));
        assert!(orient_2d(&points, |l, i| l[i], 2, 0, 1));
        assert!(!orient_2d(&points, |l, i| l[i], 2, 1, 0));
        assert_eq!(orient_2d_case(&points, |l, i| l[i], 0, 1, 2), case);
    }

    #[test_case([[0.0, 0.0, 0.0], [0.0, 0.0, 1.0], [0.0, 1.0, 0.0], [1.0, 0.0, 0.0]], [4,4,4,4] ; "General")]
    #[test_case([[0.0, 0.0, 0.0], [1.0, 1.0, 1.0], [3.0, 4.0, 5.0], [2.0, 3.0, 4.0]], [3,4,4,4] ; "Coplanar")]
    #[test_case([[0.0, 0.0, 0.0], [1.0, 1.0, 1.0], [2.0, 2.0, 4.0], [3.0, 3.0, 5.0]], [2,4,4,4] ; "Coplanar, pj pk pl @ xy collinear")]
    #[test_case([[1.0, 0.0, 0.0], [1.0, 1.0, 1.0], [1.0, 4.0, 2.0], [1.0, 5.0, 3.0]], [1,4,4,4] ; "Coplanar, pj.x = pk.x = pl.x or pj pk pl collinear")]
    #[test_case([[0.0, 0.0, 0.0], [1.0, 2.0, 3.0], [2.0, 3.0, 4.0], [3.0, 4.0, 5.0]], [3,3,4,4] ; "pj pk pl collinear")]
    #[test_case([[0.0, 0.0, 0.0], [1.0, 1.0, 3.0], [3.0, 3.0, 5.0], [2.0, 2.0, 4.0]], [2,3,4,4] ; "pj pk pl collinear, pi pk pl @ xy collinear")]
    #[test_case([[0.0, 0.0, 0.0], [0.0, 1.0, 3.0], [0.0, 2.0, 4.0], [0.0, 3.0, 5.0]], [1,3,4,4] ; "pj pk pl collinear, pi pk pl @ xy collinear, pk.x = pl.x")]
    #[test_case([[1.0, 0.0, 0.0], [0.0, 2.0, 3.0], [0.0, 2.0, 5.0], [0.0, 2.0, 4.0]], [2,2,4,4] ; "pj pk pl collinear, pi pk pl @ xy collinear, pk.xy = pl.xy")]
    #[test_case([[0.0, 0.0, 0.0], [0.0, 2.0, 3.0], [0.0, 2.0, 4.0], [0.0, 2.0, 3.0]], [1,2,4,4] ; "pj pk pl collinear, pi.x = pk.x = pl.x or pi pk pl collinear, pk.xy = pl.xy")]
    //                                                                              , [1,1,4,4] ; "pk = pl and pi pk pl @ yz not collinear is impossible
    #[test_case([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [2.0, 1.0, 0.0], [2.0, 1.0, 0.0]], [3,3,3,4] ; "pk = pl")]
    #[test_case([[0.0, 0.0, 0.0], [1.0, 1.0, 0.0], [2.0, 2.0, 0.0], [2.0, 2.0, 0.0]], [2,3,3,4] ; "pk = pl, pi pj pk @ xy collinear")]
    #[test_case([[0.0, 0.0, 0.0], [0.0, 2.0, 0.0], [0.0, 1.0, 0.0], [0.0, 1.0, 0.0]], [1,3,3,4] ; "pk = pl, pi pj pk @ xy collinear, pj.x = pk.x")]
    #[test_case([[1.0, 0.0, 0.0], [0.0, 2.0, 0.0], [0.0, 2.0, 0.0], [0.0, 2.0, 0.0]], [2,2,3,4] ; "pk = pl, pi pj pk @ xy collinear, pj.xy = pk.xy")]
    #[test_case([[0.0, 0.0, 0.0], [0.0, 2.0, 0.0], [0.0, 2.0, 0.0], [0.0, 2.0, 0.0]], [1,2,3,4] ; "pk = pl, pi pj pk @ xy collinear, pj.xy = pk.xy, pi.x = pk.x")]
    fn test_orient_3d(points: [[f64; 3]; 4], case: [usize; 4]) {
        let points = points
            .iter()
            .copied()
            .map(Vector3::from)
            .collect::<Vec<_>>();
        // Trusting the insertion sort now
        assert!(orient_3d(&points, |l, i| l[i], 0, 1, 2, 3));
        assert!(!orient_3d(&points, |l, i| l[i], 3, 2, 0, 1));
        assert_eq!(orient_3d_case(&points, |l, i| l[i], 0, 1, 2, 3), case);
    }

    #[test_case([[0.0, 0.0], [0.0, 2.0], [2.0, 2.0], [1.0, 1.0]], [4,4,4,4] ; "General")]
    #[test_case([[1.0, 0.0], [3.0, 1.0], [2.0, 3.0], [0.0, 2.0]], [3,4,4,4] ; "Cocircular")]
    #[test_case([[0.0, 0.0], [1.0, 1.0], [3.0, 3.0], [2.0, 2.0]], [2,4,4,4] ; "Cocircular, pj pk pl collinear")]
    #[test_case([[1.0, 0.0], [1.0, 5.0], [1.0, 1.0], [1.0, 4.0]], [1,4,4,4] ; "Cocircular, pj.x = pk.x = pl.x or 2 of pj pk pl equal")]
    #[test_case([[1.0, 1.0], [0.0, 0.0], [0.0, 0.0], [2.0, 3.0]], [3,3,4,4] ; "2 of pj pk pl equal")]
    #[test_case([[2.0, 2.0], [1.0, 1.0], [1.0, 1.0], [0.0, 0.0]], [2,3,4,4] ; "2 of pj pk pl equal, pi pk pl collinear")]
    #[test_case([[1.0, 2.0], [1.0, 0.0], [1.0, 0.0], [1.0, 1.0]], [1,3,4,4] ; "2 of pj pk pl equal, pi.x = pk.x = pl.x or pk = pl")]
    //                                                          , [2,2,4,4] ; "pk = pl and pi pk pl @ m2-x not collinear is impossible
    //                                                          , [1,2,4,4] ; "pk = pl and pk.m2 != pl.m2 is impossible
    //                                                          , [1,1,4,4] ; "pk = pl and pi pk pl @ y-m2 not collinear is impossible
    #[test_case([[0.0, 0.0], [1.0, 0.0], [2.0, 1.0], [2.0, 1.0]], [3,3,3,4] ; "pk = pl")]
    #[test_case([[0.0, 0.0], [1.0, 1.0], [2.0, 2.0], [2.0, 2.0]], [2,3,3,4] ; "pk = pl, pi pj pk collinear")]
    #[test_case([[0.0, 0.0], [0.0, 2.0], [0.0, 1.0], [0.0, 1.0]], [1,3,3,4] ; "pk = pl, pi.x = pj.x = pk.x or pj = pk")]
    #[test_case([[1.0, 0.0], [0.0, 2.0], [0.0, 2.0], [0.0, 2.0]], [2,2,3,4] ; "pj = pk = pl")]
    #[test_case([[0.0, 0.0], [0.0, 2.0], [0.0, 2.0], [0.0, 2.0]], [1,2,3,4] ; "pj = pk = pl, pi.x = pk.x")]
    fn test_in_circle(points: [[f64; 2]; 4], case: [usize; 4]) {
        let points = points
            .iter()
            .copied()
            .map(Vector2::from)
            .collect::<Vec<_>>();
        // Trusting the insertion sort now
        assert!(in_circle(&points, |l, i| l[i], 0, 1, 2, 3));
        assert!(in_circle(&points, |l, i| l[i], 0, 2, 1, 3));
        assert!(in_circle(&points, |l, i| l[i], 1, 2, 0, 3));
        assert!(in_circle(&points, |l, i| l[i], 1, 0, 2, 3));
        assert!(in_circle(&points, |l, i| l[i], 2, 0, 1, 3));
        assert!(in_circle(&points, |l, i| l[i], 2, 1, 0, 3));
        assert!(
            (in_circle(&points, |l, i| l[i], 0, 1, 2, 3)
                == in_circle(&points, |l, i| l[i], 0, 1, 3, 2))
                == (orient_2d(&points, |l, i| l[i], 0, 1, 3)
                    != orient_2d(&points, |l, i| l[i], 0, 1, 2))
        );
        assert_eq!(in_circle_case(&points, |l, i| l[i], 0, 1, 2, 3), case);
    }

    // Taking integers to shorten things
    #[test_case([[0,0,0], [4,0,0], [0,4,0], [0,0,4], [1,1,1]], [5,5,5,5,5] ; "General")]
    #[test_case([[0,0,0], [2,0,0], [2,2,2], [2,2,0], [0,2,2]], [4,5,5,5,5] ; "Cospherical")]
    #[test_case([[0,0,0], [1,1,1], [2,2,2], [2,4,4], [2,3,3]], [3,5,5,5,5] ; "Cospherical, pj pk pl pm coplanar")]
    #[test_case([[2,2,2], [0,0,2], [2,2,0], [0,0,1], [1,1,0]], [2,5,5,5,5] ; "Cospherical, pj pk pl pm coplanar, pj pk pl pm @ xy-m3 coplanar")]
    #[test_case([[0,1,0], [0,2,0], [0,2,1], [0,1,2], [0,1,1]], [1,5,5,5,5] ; "Cospherical, pj pk pl pm coplanar, pj pk pl pm @ xy-m3 and zx-m3 coplanar")]
    #[test_case([[0,0,0], [1,0,0], [1,0,0], [0,0,1], [0,1,0]], [4,4,5,5,5] ; "pj pk pl pm cocircular")]
    #[test_case([[2,0,0], [2,0,0], [2,0,0], [2,1,0], [1,0,0]], [3,4,5,5,5] ; "pj pk pl pm cocircular, pi pk pl pm coplanar")]
    #[test_case([[0,0,0], [0,0,0], [1,0,0], [0,0,0], [0,0,1]], [2,4,5,5,5] ; "pj pk pl pm cocircular, pi pk pl pm coplanar, pk pl pm @ xy collinear")]
    #[test_case([[0,0,0], [0,0,0], [0,0,0], [0,1,0], [0,0,1]], [1,4,5,5,5] ; "pj pk pl pm cocircular, pi pk pl pm coplanar, pk pl pm @ xy and zx collinear")]
    #[test_case([[2,0,2], [1,2,1], [1,0,1], [1,2,1], [1,1,1]], [3,3,5,5,5] ; "pj pk pl pm cocircular, pi pk pl pm coplanar, pk pl pm collinear")]
    #[test_case([[2,0,0], [2,0,1], [2,0,1], [0,0,1], [1,0,1]], [2,3,5,5,5] ; "pj pk pl pm cocircular, pi pk pl pm coplanar, pk pl pm collinear, pi pk pl pm @ xy-m3 coplanar")]
    #[test_case([[0,1,0], [0,1,1], [0,2,1], [0,0,1], [0,1,1]], [1,3,5,5,5] ; "pj pk pl pm cocircular, pi pk pl pm coplanar, pk pl pm collinear, pi pk pl pm @ xy-m3 coplanar, pk pl pm @ x-m3 collinear")]
    #[test_case([[0,0,2], [1,0,2], [1,0,0], [1,0,2], [1,0,1]], [2,2,5,5,5] ; "pj pk pl pm cocircular, pi pk pl pm coplanar, pk pl pm collinear, pi pk pl pm @ xy-m3 coplanar, pk pl pm @ x-m3 and y-m3 collinear")]
    #[test_case([[0,1,2], [0,2,1], [0,2,2], [0,2,0], [0,2,1]], [1,2,5,5,5] ; "pj pk pl pm cocircular, pi pk pl pm coplanar, pk pl pm collinear, pi pk pl pm @ xy-m3 and zx-m3 coplanar, pk pl pm @ x-m3 and y-m3 collinear")]
    //                                                         [1,1,5,5,5] ; "2 of pk pl pm are equal and pi pk pl pm @ yz-m3 are not coplanar is impossible
    #[test_case([[0,0,0], [1,0,0], [0,1,0], [0,1,0], [0,0,1]], [4,4,4,5,5] ; "2 of pk pl pm are equal")]
    #[test_case([[0,2,0], [0,2,0], [1,2,0], [1,2,0], [0,1,0]], [3,4,4,5,5] ; "2 of pk pl pm are equal, pi pj pl pm coplanar")]
    #[test_case([[0,0,0], [0,0,0], [1,0,0], [1,0,0], [0,0,1]], [2,4,4,5,5] ; "2 of pk pl pm are equal, pi pj pl pm coplanar, pj pl pm @ xy collinear")]
    #[test_case([[0,0,0], [0,0,0], [0,1,0], [0,0,1], [0,1,0]], [1,4,4,5,5] ; "2 of pk pl pm are equal, pi pj pl pm coplanar, pj pl pm @ xy and zx collinear")]
    #[test_case([[0,0,0], [1,0,0], [0,1,0], [1,0,0], [0,1,0]], [3,3,4,5,5] ; "2 of pk pl pm are equal, pi pj pl pm coplanar, pj pl pm collinear")]
    #[test_case([[0,0,0], [0,0,0], [0,0,0], [0,0,0], [1,0,0]], [2,3,4,5,5] ; "2 of pk pl pm are equal, pi pj pl pm coplanar, pj pl pm collinear, pi pl pm @ xy collinear")]
    #[test_case([[0,2,0], [0,2,0], [0,2,0], [0,2,0], [0,1,0]], [1,3,4,5,5] ; "2 of pk pl pm are equal, pi pj pl pm coplanar, pj pl pm collinear, pi pl pm @ xy collinear, pl.x = pm.x")]
    #[test_case([[0,0,0], [1,0,0], [1,0,0], [1,0,1], [1,0,0]], [2,2,4,5,5] ; "2 of pk pl pm are equal, pi pj pl pm coplanar, pj pl pm collinear, pi pl pm @ xy collinear, pl.xy = pm.xy")]
    #[test_case([[0,0,0], [0,0,0], [0,0,0], [0,0,0], [0,0,1]], [1,2,4,5,5] ; "2 of pk pl pm are equal, pi pj pl pm coplanar, pj pl pm collinear, pi pl pm @ xy and zx collinear, pl.xy = pm.xy")]
    //                                                         [1,1,4,5,5] ; "pl = pm and some determinant involving pl and pm is not 0 is impossible
    //                                                         [3,3,3,5,5] ; "pl = pm and some determinant involving pl and pm is not 0 is impossible
    //                                                         [2,3,3,5,5] ; "pl = pm and some determinant involving pl and pm is not 0 is impossible
    //                                                         [1,3,3,5,5] ; "pl = pm and some determinant involving pl and pm is not 0 is impossible
    //                                                         [2,2,3,5,5] ; "pl = pm and some determinant involving pl and pm is not 0 is impossible
    //                                                         [1,2,3,5,5] ; "pl = pm and some determinant involving pl and pm is not 0 is impossible
    //                                                         [1,1,3,5,5] ; "pl = pm and some determinant involving pl and pm is not 0 is impossible
    //                                                         [2,2,2,5,5] ; "pl = pm and some determinant involving pl and pm is not 0 is impossible
    //                                                         [1,2,2,5,5] ; "pl = pm and some determinant involving pl and pm is not 0 is impossible
    //                                                         [1,1,2,5,5] ; "pl = pm and some determinant involving pl and pm is not 0 is impossible
    //                                                         [1,1,1,5,5] ; "pl = pm and some determinant involving pl and pm is not 0 is impossible
    #[test_case([[0,0,0], [1,1,1], [0,1,0], [1,0,0], [1,0,0]], [4,4,4,4,5] ; "pl = pm")]
    #[test_case([[0,0,0], [1,1,1], [3,4,5], [2,3,4], [2,3,4]], [3,4,4,4,5] ; "pl = pm, coplanar")]
    #[test_case([[0,0,0], [1,1,1], [2,2,4], [3,3,5], [3,3,5]], [2,4,4,4,5] ; "pl = pm, coplanar, pj pk pl @ xy collinear")]
    #[test_case([[1,0,0], [1,1,1], [1,4,2], [1,5,3], [1,5,3]], [1,4,4,4,5] ; "pl = pm, coplanar, pj pk pl @ xy and yz collinear")]
    #[test_case([[0,0,0], [1,2,3], [2,3,4], [3,4,5], [3,4,5]], [3,3,4,4,5] ; "pl = pm, pj pk pl collinear")]
    #[test_case([[0,0,0], [1,1,3], [3,3,5], [2,2,4], [2,2,4]], [2,3,4,4,5] ; "pl = pm, pj pk pl collinear, pi pk pl @ xy collinear")]
    #[test_case([[0,0,0], [0,1,3], [0,2,4], [0,3,5], [0,3,5]], [1,3,4,4,5] ; "pl = pm, pj pk pl collinear, pi pk pl @ xy collinear, pk.x = pl.x")]
    #[test_case([[1,0,0], [0,2,3], [0,2,5], [0,2,4], [0,2,4]], [2,2,4,4,5] ; "pl = pm, pj pk pl collinear, pi pk pl @ xy collinear, pk.xy = pl.xy")]
    #[test_case([[0,0,0], [0,2,3], [0,2,4], [0,2,3], [0,2,3]], [1,2,4,4,5] ; "pl = pm, pj pk pl collinear, pi pk pl @ xy and zx collinear, pk.xy = pl.xy")]
    //                                                       , [1,1,4,4,5] ; "pk = pl = pm and pi pk pl @ yz not collinear is impossible
    #[test_case([[0,0,0], [1,0,0], [2,1,0], [2,1,0], [2,1,0]], [3,3,3,4,5] ; "pk = pl = pm")]
    #[test_case([[0,0,0], [1,1,0], [2,2,0], [2,2,0], [2,2,0]], [2,3,3,4,5] ; "pk = pl = pm, pi pj pk @ xy collinear")]
    #[test_case([[0,0,0], [0,2,0], [0,1,0], [0,1,0], [0,1,0]], [1,3,3,4,5] ; "pk = pl = pm, pi pj pk @ xy collinear, pj.x = pk.x")]
    #[test_case([[1,0,0], [0,2,0], [0,2,0], [0,2,0], [0,2,0]], [2,2,3,4,5] ; "pk = pl = pm, pi pj pk @ xy collinear, pj.xy = pk.xy")]
    #[test_case([[0,0,0], [0,2,0], [0,2,0], [0,2,0], [0,2,0]], [1,2,3,4,5] ; "pk = pl = pm, pi pj pk @ xy collinear, pj.xy = pk.xy, pi.x = pk.x")]
    fn test_in_sphere(points: [[i32; 3]; 5], case: [usize; 5]) {
        let points = points
            .iter()
            .copied()
            .map(|[x, y, z]| Vector3::new(x as f64, y as f64, z as f64))
            .collect::<Vec<_>>();
        // Trusting the insertion sort now
        assert!(in_sphere(&points, |l, i| l[i], 0, 1, 2, 3, 4));
        assert!(in_sphere(&points, |l, i| l[i], 0, 2, 1, 3, 4));
        assert!(in_sphere(&points, |l, i| l[i], 1, 2, 0, 3, 4));
        assert!(in_sphere(&points, |l, i| l[i], 1, 3, 0, 2, 4));
        assert!(in_sphere(&points, |l, i| l[i], 2, 3, 0, 1, 4));
        assert!(in_sphere(&points, |l, i| l[i], 2, 3, 1, 0, 4));
        assert!(
            (in_sphere(&points, |l, i| l[i], 0, 1, 2, 3, 4)
                == in_sphere(&points, |l, i| l[i], 0, 1, 2, 4, 3))
                == (orient_3d(&points, |l, i| l[i], 0, 1, 2, 3)
                    != orient_3d(&points, |l, i| l[i], 0, 1, 2, 4))
        );
        assert_eq!(in_sphere_case(&points, |l, i| l[i], 0, 1, 2, 3, 4), case);
    }
}
