extern crate proc_macro;
#[macro_use]
extern crate quote;

use fnv::FnvHashMap;
use proc_macro::TokenStream;
use syn::{Ident, Token};
use syn::parse::{Parse, ParseStream, Result};
use itertools::Itertools;
use permutator::Combination;
use std::{collections::HashSet, fmt::{self, Display, Formatter}};
use std::iter::{once, repeat};
use proc_macro2::TokenStream as TokenStream2;

struct InHypersphere {
    /// The list to index on
    list: Ident,
    /// The indexing function
    index_fn: Ident,
    /// The list of indexes
    indexes: Vec<Ident>,
}

impl Parse for InHypersphere {
    fn parse(input: ParseStream) -> Result<Self> {
        let list: Ident = input.parse()?;
        input.parse::<Token![,]>()?;
        let index_fn: Ident = input.parse()?;
        input.parse::<Token![,]>()?;
        let indexes = input.parse_terminated::<Ident, Token![,]>(Ident::parse)?;
        
        Ok(InHypersphere {
            list,
            index_fn,
            indexes: indexes.into_iter().collect()
        })
    }
}

/// Sub-determinant of the original matrix.
/// Row the last is implicity included.
/// Column the last (the column of 1's) is implicity included.
#[derive(Clone, Debug, Default, PartialEq, Eq, Hash, PartialOrd, Ord)]
struct Determinant {
    rows: Vec<usize>,
    cols: Vec<usize>,
}

impl Determinant {
    fn new(rows: Vec<usize>, cols: Vec<usize>) -> Self {
        Self { rows, cols }
    }

    fn nonzero(self, zero_dets: &mut HashSet<Determinant>) -> Option<Self> {
        if zero_dets.contains(&self) {
            return None;
        };

        // Smaller determinants
        for i in 1..self.cols.len() {
            if self.rows.combination(i).any(|combo_r|
                self.cols.combination(i).all(|combo_c|
                    zero_dets.contains(&Determinant::new(combo_r.iter().copied().copied().collect(),
                        combo_c.into_iter().copied().collect()))))
            {
                // Determinant is 0 because a whole row/rows of subdeterminants are 0
                return None;
            }
        }

        Some(self)
    }

    /// To be called after prepare_dets_for_cases
    fn vector_tokens(&self, points: &[Ident]) -> TokenStream2 {
        let dim = points.len() - 2;
        let mut cols = self.cols.clone();

        // Magnitude column; replace with missing coordinates
        if cols[cols.len() - 1] == dim {
            cols.pop();
            cols.extend((0..dim).filter(|i| !self.cols.contains(i)));
        }

        let vector = format_ident!("Vector{}", cols.len());

        self.rows.iter().map(|r| {
            let point = &points[*r];
            let mut coords = cols.iter().map(|c| {
                quote! { #point[#c], }
            }).collect::<Vec<_>>();

            if coords.len() > 1 {
                let coords = coords.into_iter().collect::<TokenStream2>();
                quote! { nalgebra::#vector::new(#coords), }
            } else {
                coords.pop().unwrap()
            }
        }).collect()
    }

    fn to_grid(&self, indexes: &[Ident]) -> Vec<String> {
        let coords = "xyzw".chars().collect::<Vec<_>>();
        let mut lines = vec![];
        for row in self.rows.iter().copied().chain(once(indexes.len() - 1)) {
            let mut line = "│ ".to_string();

            for col in self.cols.iter().copied().chain(once(indexes.len() - 1)) {
                if col == indexes.len() - 1 {
                    line += "1 ";
                } else if col == indexes.len() - 2 {
                    line += &(0..indexes.len() - 2).map(|i| format!("{}{}²", indexes[row], coords[i])).join("+");
                    line += "  ";
                } else {
                    line += &format!("{}{}  ", indexes[row], coords[col]);
                }
            }

            lines.push(line + "│");
        }

        //let pad = repeat(" ").take(lines[0].chars().count() - 2).collect::<String>();
        //lines.insert(0, format!("│{}│", pad));
        //lines.push(format!("│{}│", pad));
        lines
    }
}

#[derive(Clone, Debug, PartialEq, Eq, Hash)]
struct Term {
    const_mult: i32,
    /// Says location of term to multiply by.
    var_mult: Option<[usize; 2]>,
    det: Determinant,
}

impl Term {
    fn new(const_mult: i32, var_mult: Option<[usize; 2]>, det: Determinant) -> Self {
        Self { const_mult, var_mult, det }
    }

    fn nonzero(mut self, zero_dets: &mut HashSet<Determinant>) -> Option<Self> {
        if let Some(det) = std::mem::take(&mut self.det).nonzero(zero_dets) {
            self.det = det;
            Some(self)
        } else {
            None
        }
    }

    fn to_grid(&self, indexes: &[Ident]) -> Vec<String> {
        let coords = "xyzw".chars().collect::<Vec<_>>();
        let mut lines = self.det.to_grid(indexes);

        let mut coeff = if self.const_mult >= 0 {"+ "} else {"- "}.to_owned();
        if self.const_mult.abs() != 1 {
            coeff += &self.const_mult.abs().to_string();
        }
        if let Some([r, c]) = self.var_mult {
            coeff += &format!("{}{}", indexes[r], coords[c]);
        }

        let mid = (lines.len() - 1) / 2;
        let pad = repeat(" ").take(coeff.chars().count()).collect::<String>();
        lines[mid] = coeff + &lines[mid];
        for (i, line) in lines.iter_mut().enumerate() {
            if i != mid {
                *line = pad.clone() + line;
            }
        }
        
        lines
    }
}

#[derive(Clone, Debug, Default)]
struct TermSum {
    terms: Vec<Term>,
}

impl TermSum {
    fn new() -> Self {
        Self::default()
    }

    fn without_zero_dets(mut self, dim: usize, zero_dets: &mut HashSet<Determinant>) -> Option<Self> {
        self.terms = self.terms.into_iter().flat_map(|t| t.nonzero(zero_dets)).collect::<Vec<_>>();

        if self.terms.len() == 1 && self.terms[0].var_mult.is_none() {
            let det = &self.terms[0].det;
            zero_dets.insert(det.clone());

            // Special case: coordinates equal, so the magnitudes do as well.
            if det.cols.len() == 1 && (0..dim).all(|i| 
                zero_dets.contains(&Determinant::new(vec![det.rows[0]], vec![i])))
            {
                zero_dets.insert(Determinant::new(vec![det.rows[0]], vec![dim]));
            }
        }

        if self.terms.is_empty() { None } else { Some(self) }
    }

    fn prepare_dets_for_cases(&mut self, dim: usize) {
        for term in &mut self.terms {
            // For convenience of including the last point
            term.det.rows.push(dim + 1);

            if term.const_mult < 0 {
                term.const_mult *= -1;
                let n = term.det.rows.len();
                term.det.rows.swap(n - 2, n - 1);
            }
        }
    }

    fn case(mut self, points: &[Ident]) -> TokenStream2 {
        let coords = "xyzw".chars().collect::<Vec<_>>();
        let dim = points.len() - 2;
        self.prepare_dets_for_cases(dim);

        if self.terms.len() == 1 && self.terms[0].det.cols.len() == dim + 1 {
            assert_eq!(self.terms[0].const_mult, 1);
            assert_eq!(self.terms[0].var_mult, None);

            if dim == 2 {
                let [i, j, k, l] = [&points[0], &points[1], &points[2], &points[3]];
                quote! {
                    let val = rg::in_circle(#i, #j, #k, #l);
                    if val != 0.0 {
                        return (val > 0.0) != odd;
                    }
                }
            } else if dim == 3 {
                let [i, j, k, l, m] = [&points[0], &points[1], &points[2], &points[3], &points[4]];
                quote! {
                    let val = rg::in_sphere(#i, #j, #k, #l, #m);
                    if val != 0.0 {
                        return (val > 0.0) != odd;
                    }
                }
            } else {
                panic!("Unsupported # of dimensions: {}", dim)
            }
        } else if self.terms.len() == 1 && self.terms[0].det.cols.last() == Some(&dim) {
            assert_eq!(self.terms[0].const_mult, 1);
            assert_eq!(self.terms[0].var_mult, None);

            let det = self.terms[0].det.vector_tokens(points);
            let func = if self.terms[0].det.cols.len() == 1 {
                format_ident!("magnitude_cmp_{}d", dim)
            } else {
                format_ident!(
                    "sign_det_{}{}",
                    coords[..self.terms[0].det.cols.len() - 1].iter().map(|c| c.to_string() + "_").join(""),
                    coords[..dim].iter().map(|c| c.to_string() + "2").join(""),
                )
            };
            quote! {
                let val = rg::#func(#det);
                if val != 0.0 {
                    return (val > 0.0) != odd;
                }
            }
        } else if self.terms.len() == 1 {
            assert_eq!(self.terms[0].const_mult, 1);
            assert_eq!(self.terms[0].var_mult, None);
            
            if self.terms[0].det.cols.len() == 0 {
                quote! { !odd }
            } else if self.terms[0].det.cols.len() == 1 {
                let coord = self.terms[0].det.cols[0];
                let p1 = &points[self.terms[0].det.rows[0]];
                let p2 = &points[self.terms[0].det.rows[1]];
                quote! {
                    if #p1[#coord] != #p2[#coord] {
                        return (#p1[#coord] > #p2[#coord]) != odd;
                    }
                }
            } else {
                let det = self.terms[0].det.vector_tokens(points);
                let func = format_ident!("orient_{}d", self.terms[0].det.cols.len());
                quote! {
                    let val = rg::#func(#det);
                    if val != 0.0 {
                        return (val > 0.0) != odd;
                    }
                }
            }
        } else if self.terms.len() == 2 && self.terms[0].var_mult.is_none() {
            assert_eq!(self.terms[0].const_mult, 1);
            assert_eq!(*self.terms[0].det.cols.last().unwrap(), dim);
            assert_eq!(self.terms[1].const_mult, 2);
            assert!(self.terms[1].var_mult.is_some());
            assert_ne!(*self.terms[1].det.cols.last().unwrap(), dim);

            let det1 = self.terms[0].det.vector_tokens(points);
            let det2 = self.terms[1].det.vector_tokens(points);
            let mult = &points[self.terms[1].var_mult.unwrap()[0]];
            let mult_coord = self.terms[1].var_mult.unwrap()[1];
            let func = format_ident!(
                "sign_det_{}{}_plus_2x_det_{}",
                coords[..self.terms[0].det.cols.len() - 1].iter().map(|c| c.to_string() + "_").join(""),
                coords[..dim].iter().map(|c| c.to_string() + "2").join(""),
                coords[..self.terms[1].det.cols.len()].iter().join("_"),
            );
            quote! { 
                let val = rg::#func(#det1 #mult[#mult_coord], #det2);
                if val != 0.0 {
                    return (val > 0.0) != odd;
                }
            }
        } else if self.terms.len() == 2 {
            assert_eq!(self.terms[0].const_mult, 2);
            assert_ne!(self.terms[0].det.cols.last(), Some(&dim));
            assert_eq!(self.terms[1].const_mult, 2);
            assert!(self.terms[1].var_mult.is_some());
            assert_eq!(self.terms[0].det, self.terms[1].det);

            let mult1 = &points[self.terms[0].var_mult.unwrap()[0]];
            let mult1_coord = self.terms[0].var_mult.unwrap()[1];
            let mult2 = &points[self.terms[1].var_mult.unwrap()[0]];
            let mult2_coord = self.terms[1].var_mult.unwrap()[1];
            
            let inner = if self.terms[0].det.cols.len() == 0 {
                quote! { return negate == odd; }
            } else if self.terms[0].det.cols.len() == 1 {
                let coord = self.terms[0].det.cols[0];
                let p1 = &points[self.terms[0].det.rows[0]];
                let p2 = &points[self.terms[0].det.rows[1]];
                quote! {
                    if #p1[#coord] != #p2[#coord] {
                        return (#p1[#coord] > #p2[#coord]) != (negate != odd);
                    }
                }
            } else {
                let det = self.terms[0].det.vector_tokens(points);
                let func = format_ident!("orient_{}d", self.terms[0].det.cols.len());
                quote! {
                    let val = rg::#func(#det);
                    if val != 0.0 {
                        return (val > 0.0) != (negate != odd);
                    }
                }
            };

            quote! {
                if #mult1[#mult1_coord] != -#mult2[#mult2_coord] {
                    let negate = #mult1[#mult1_coord] < -#mult2[#mult2_coord];
                    #inner
                }
            }
        } else {
            panic!("Unsupported determinant: {}", self.to_grid(points).join("\n"))
        }
    }

    fn to_grid(&self, indexes: &[Ident]) -> Vec<String> {
        let mut lines = self.terms[0].to_grid(indexes);
        for term in &self.terms[1..] {
            for (i, line) in term.to_grid(indexes).into_iter().enumerate() {
                lines[i] += &format!(" {}", line);
            }
        }
        lines
    }
}

/// An ε-factor, represented as an exponent of ε.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
struct EFactor(u64);

impl EFactor {
    fn new(dim: usize, coords: impl IntoIterator<Item = [usize; 2]>) -> Self {
        Self(coords.into_iter().map(|[r, c]| 3u64.pow((dim * r + dim - 1 - c) as u32)).sum())
    }

    fn to_repr(mut self, indexes: &[Ident]) -> String {
        let coords = "xyzw".chars().collect::<Vec<_>>();
        let mut res = String::new();

        for index in indexes {
            for c in 0..indexes.len() - 2 {
                let rem = self.0 % 3;
                self.0 /= 3;

                if rem > 0 {
                    if !res.is_empty() {
                        res += "·";
                    }
                    res += &format!("ε{}{}", index, coords[indexes.len() - 3 - c]);
                }
                if rem == 2 {
                    res += "²";
                }
            }
        }

        res
    }
}

impl Display for EFactor {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        let mut res = String::new();
        let mut num = self.0;

        while num > 0 {
            res += &(num % 3).to_string();
            num /= 3;
        }

        res = res.chars().rev().collect();
        f.pad(&res)
    }
}

fn terms(dim: usize) -> Vec<(EFactor, Term)> {
    let mut terms = vec![];

    // The biggest relevant ε-factor.
    let big_e = EFactor::new(dim, (0..dim - 1).map(|i| [i, i]).chain(vec![[dim - 1, dim - 1], [dim - 1, dim - 1], [dim, dim - 1]]));

    let all = (0..=dim).collect::<Vec<_>>();

    // General term
    terms.push((EFactor::new(dim, vec![]), Term::new(1, None, Determinant::new(all.clone(), all.clone()))));

    // Degenerate terms
    let mut rows = all.clone();
    let mut cols = all.clone();
    let mut e_factors = vec![];
    for i in 1..=dim + 1 {
        let mut remove = vec![0; 2 * i];

        while remove[0] <= dim - (i - 1) {
            // Trying not to have a million allocations here
            rows.clear();
            rows.extend(all.iter().copied());
            cols.clear();
            cols.extend(all.iter().copied());
            e_factors.clear();

            let mut mult = 1;
            for rc in remove.chunks_exact(2) {
                let er = rows.remove(rc[0]);
                let ec = cols.remove(rc[1]);
                if (er + ec) % 2 == 1 {
                    mult *= -1;
                }
                e_factors.push([er, ec]);
            }

            let det = Determinant::new(rows.clone(), cols.clone());

            // Column dim is the magnitude column, so do special things with it.
            // For example, (x + εx)² + (y + εy)² expands to
            // (x² + y²) + εx·2x + εx² + εy·2y + εy²
            if let Some(mag_r) = e_factors.iter().position(|[_, c]| *c == dim).map(|i| e_factors.remove(i)[0]) {
                for j in 0..dim {
                    let factor = EFactor::new(dim, e_factors.iter().copied().chain(once([mag_r, j])));
                    if factor <= big_e {
                        terms.push((factor, Term::new(mult * 2, Some([mag_r, j]), det.clone())));
                    }

                    let factor = EFactor::new(dim, e_factors.iter().copied().chain(repeat([mag_r, j]).take(2)));
                    if factor <= big_e {
                        terms.push((factor, Term::new(mult, None, det.clone())));
                    }
                }
            } else {
                let factor = EFactor::new(dim, e_factors.drain(..));
                if factor <= big_e {
                    terms.push((factor, Term::new(mult, None, det)));
                }
            }

            // Count in base factorial to iterate through permutations
            // Row index shouldn't decrease so permutations aren't repeated.
            let mut j = 2 * i - 1;
            while {
                remove[j] += 1;
                if j % 2 == 0 && remove[j] <= dim - (i - 1) {
                    let row = remove[j];
                    for n in remove[j + 2..].iter_mut().step_by(2) {
                        *n = row;
                    }
                }

                remove[j] > dim - if j % 2 == 0 {i - 1} else {j / 2} && j > 0
            } {
                if j % 2 == 0 {
                    let row = remove[j - 2];
                    for n in remove[j..].iter_mut().step_by(2) {
                        *n = row;
                    }
                } else {
                    remove[j] = 0;
                };

                j -= 1;
            }
        }
    }

    terms
}

// Ordered by ε-factor exponent
fn term_sums(dim: usize) -> Vec<(EFactor, TermSum)> {
    let mut sums = FnvHashMap::default();

    for (e, term) in terms(dim) {
        sums.entry(e).or_insert(TermSum::new()).terms.push(term);
    }

    let mut sums = sums.into_iter().collect::<Vec<_>>();
    sums.sort_by_key(|(e, _)| *e);
    sums
}

fn fn_body(h: InHypersphere, sums: Vec<(EFactor, TermSum)>) -> TokenStream2 {
    let list = h.list;
    let index_fn = h.index_fn;
    let dim = h.indexes.len() - 2;

    let orient = format_ident!("orient_{}d", dim);
    let sorted = format_ident!("sorted_{}", h.indexes.len());
    let index_short_seq = h.indexes[..h.indexes.len() - 1].iter().map(|index| quote! {#index,})
        .collect::<TokenStream2>();
    let index_seq = h.indexes.iter().map(|index| quote!{#index,}).collect::<TokenStream2>();

    let points = h.indexes.iter().map(|index| format_ident!("p{}", index)).collect::<Vec<_>>();
    let indexing_seq = h.indexes.iter().zip(points.iter()).map(|(index, point)| quote! {
        let #point = #index_fn(#list, #index);
    }).collect::<TokenStream2>();

    let mut zero_dets = HashSet::new();
    let cases = sums.into_iter()
        .flat_map(|(e, sum)| sum.without_zero_dets(dim, &mut zero_dets).map(|sum| (e, sum)))
        .map(|(_, sum)| sum.case(&points))
        .collect::<TokenStream2>();

    let tokens = quote! { 
        let flip = !#orient(#list, #index_fn.clone(), #index_short_seq);
        let ([#index_seq], odd) = #sorted([#index_seq]);
        let odd = odd != flip;

        #indexing_seq

        #cases
    };

    tokens
}

#[proc_macro]
pub fn generate_in_hypersphere(input: TokenStream) -> TokenStream {
    let h = syn::parse_macro_input!(input as InHypersphere);

    let sums = term_sums(h.indexes.len() - 2);
    //let mut msg = "Cases:\n```".to_owned();

    //let mut zero_dets = HashSet::new();
    //for (e, sum) in &sums {
    //    msg += &format!("{}:\n", e.to_repr(&h.indexes));

    //    if let Some(sum) = sum.clone().without_zero_dets(h.indexes.len() - 2, &mut zero_dets) {
    //        msg += &format!("{}\n", sum.to_grid(&h.indexes).into_iter().join("\n"));
    //    } else {
    //        msg += "Impossible!\n";
    //    }
    //    msg += "\n";
    //}
    //msg += "```";

    //let ident = quote::format_ident!("__test_macro_{}", h.indexes.len() - 2);
    //let stream = msg.split('\n').map(|line| quote! {
    //    #[doc = #line]
    //}).chain(once(quote! {
    //    pub fn #ident() {}
    //})).collect::<TokenStream2>();

    TokenStream::from(fn_body(h, sums))
}