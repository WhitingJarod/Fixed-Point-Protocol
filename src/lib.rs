use std::ops::{Add, Mul, Neg, Sub};

const SIGN_BIT: isize = 1 << (isize::BITS - 1);

mod doubles {
    pub const MANTISSA_DIGITS: u64 = f64::MANTISSA_DIGITS as u64 - 1;
    pub const EXP_DIGITS: u64 = 63 - MANTISSA_DIGITS;
    pub const EXP_MASK: u64 = (u64::MAX << MANTISSA_DIGITS + 1) >> 1;
    pub const MANTISSA_MASK: u64 = u64::MAX >> (64 - MANTISSA_DIGITS);
    pub const EXP_BIAS: i64 = (1 << (EXP_DIGITS - 1)) - 1;
}

// P is the number of decimal places
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub struct FixedPoint<const P: u8>(isize);

impl<const P: u8> FixedPoint<P> {
    pub fn zero() -> Self {
        #[cfg(debug_assertions)]
        {
            if P >= isize::BITS as u8 {
                panic!(
                    "Too many decimal places for FixedPoint. Maximum is {}.",
                    isize::BITS - 1
                );
            }
        }
        Self(0)
    }

    pub fn into_int(mut self) -> isize {
        let sign = self.0.is_negative();
        if sign {
            self.0 = -self.0;
        }
        self.0 >>= P;
        if sign {
            self.0 |= SIGN_BIT;
        }
        self.0
    }

    pub fn from_f64(mut value: f64) -> Self {
        if value == 0.0 {
            return FixedPoint(0);
        }
        let sign = value.is_sign_negative();
        value = value.abs();
        let bits = value.to_bits();
        let exp = ((bits & doubles::EXP_MASK) >> doubles::MANTISSA_DIGITS) as i64;
        let exp_val = if exp == 0 {
            exp - doubles::EXP_BIAS as i64 + 1
        } else {
            exp - doubles::EXP_BIAS as i64
        };
        let mut val = (bits & doubles::MANTISSA_MASK) as isize;
        // #[cfg(debug_assertions)]
        // {
        //     if P >= isize::BITS as u8 {
        //         panic!(
        //             "Too many decimal places for FixedPoint. Maximum is {}.",
        //             isize::BITS - 1
        //         );
        //     }
        //     let mut smallest_bit = doubles::MANTISSA_DIGITS as i64;
        //     for i in 0..doubles::MANTISSA_DIGITS {
        //         if val & (1 << i) != 0 {
        //             smallest_bit = i as i64;
        //             break;
        //         }
        //     }

        //     if (exp_val - (doubles::MANTISSA_DIGITS as i64 - smallest_bit)) < -(P as i64) {
        //         panic!("Value is too small to fit in FixedPoint.");
        //     }
        //     if exp_val > isize::BITS as i64 - P as i64 {
        //         let bytes = bits.to_be_bytes();
        //         let mut output = String::new();
        //         for byte in bytes.iter() {
        //             output.push_str(&format!("{:08b} ", byte));
        //         }
        //         panic!("Value is too large to fit in FixedPoint.",);
        //     }
        //     if (bits & doubles::EXP_MASK) ^ doubles::EXP_MASK == 0 {
        //         panic!("Value is NaN or infinity.")
        //     }
        // }
        if exp > 0 {
            val |= 1 << doubles::MANTISSA_DIGITS;
        }

        let sham = doubles::MANTISSA_DIGITS as i64 - exp_val - P as i64;
        #[cfg(debug_assertions)]
        {
            val = if sham < 0 {
                val.overflowing_shl((-sham) as u32).0
            } else {
                val.overflowing_shr(sham as u32).0
            };
        }
        #[cfg(not(debug_assertions))]
        {
            val >>= sham;
        }

        let mut val = val as isize;
        if sign {
            val = -val;
        }
        FixedPoint(val)
    }
}

#[cfg(debug_assertions)]
macro_rules! format_thousands {
    ($num:expr) => {{
        let s = $num.to_string();
        let mut stream = s.chars().rev().peekable();
        let mut output = String::new();
        let mut i = -1;
        while let Some(c) = stream.peek() {
            i += 1;
            if i == 3 && c != &'-' {
                i = 0;
                output.push(',');
            }
            output.push(stream.next().unwrap());
        }
        output.chars().rev().collect::<String>()
    }};
}
#[cfg(debug_assertions)]
macro_rules! panic_numbers {
    ($fmt:literal $(, $name:ident=$num:expr)*) => {{
        $(
            let $name: String = format_thousands!($num);
        )*
        let strs = [$(&$name),*];
        let width = strs.iter().fold(0, |a, b| b.len().max(a));
        panic!($fmt $(,$name=format!("{: >width$}", $name))*)
    }};
}
#[cfg(debug_assertions)]
macro_rules! format_numbers {
    ($fmt:literal $(, $name:ident=$num:expr)*) => {{
        $(
            let $name: String = format_thousands!($num);
        )*
        let strs = [$(&$name),*];
        let width = strs.iter().fold(0, |a, b| b.len().max(a));
        format!($fmt $(,$name=format!("{: >width$}", $name))*)
    }};
}
#[cfg(debug_assertions)]
macro_rules! panic_message {
    (too_many_decimals, $p:expr) => {
        panic_numbers!(
            "FixedPoint was initialized with too many decimal places for this architecture.\n\t\
                The maximum allowed is FixedPoint<{a}>\n\t\
                The value given was    FixedPoint<{b}>",
            a = isize::BITS - 1,
            b = $p
        );
    };
    (int_too_big, $p:expr, $value:expr) => {
        panic_numbers!(
            "FixedPoint was initialized with a value too large.\n\t\
                The maximum allowed value is {a}\n\t\
                The integer provided was     {b}",
            a = isize::MAX >> P as isize,
            b = $value
        );
    };
    (int_too_small, $p:expr, $value:expr) => {
        panic_numbers!(
            "FixedPoint was initialized with a value too negative.\n\t\
                The smallest allowed value is {a}\n\t\
                The integer provided was      {b}",
            a = -1 - (isize::MAX >> $p as isize),
            b = $value
        );
    };
}

macro_rules! from_int_impl {
    ($($t:ty),*) => {
        $(
            impl<const P: u8> From<$t> for FixedPoint<P> {
                fn from(value: $t) -> Self {
                    let mut value = value as isize;
                    #[cfg(debug_assertions)]
                    {
                        if P >= isize::BITS as u8 {
                            panic_message!(too_many_decimals, P);
                        }
                        if value > isize::MAX >> P as isize {
                            panic_message!(int_too_big, P, value);
                        }
                        if value < -1 - (isize::MAX >> P as isize) {
                            panic_message!(int_too_small, P, value);
                        }
                    }
                    let sign = value.is_negative();
                    value <<= P;
                    if sign {
                        value |= SIGN_BIT;
                    }
                    FixedPoint(value)
                }
            }
        )*
    }
}
macro_rules! from_uint_impl {
    ($($t:ty),*) => {
        $(
            impl<const P: u8> From<$t> for FixedPoint<P> {
                fn from(value: $t) -> Self {
                    let mut value = value as usize;
                    #[cfg(debug_assertions)]
                    {
                        if P >= isize::BITS as u8 {
                            panic_message!(too_many_decimals, P);
                        }
                        if value > isize::MAX as usize >> P as usize {
                            panic_message!(int_too_big, P, value);
                        }
                    }
                    value <<= P;
                    FixedPoint(value as isize)
                }
            }
        )*
    }
}

from_int_impl!(i8, i16, i32, i64, isize);
from_uint_impl!(u8, u16, u32, u64, usize);

// impl<const P: u8> From<isize> for FixedPoint<P> {
//     fn from(mut value: isize) -> Self {
//         #[cfg(debug_assertions)]
//         {
//             if P >= isize::BITS as u8 {
//                 panic!(
//                     "Too many decimal places for FixedPoint. Maximum is {}.",
//                     isize::BITS - 1
//                 );
//             }
//             if value.abs() > isize::MAX >> P {
//                 panic!("Value is too large to fit in FixedPoint.");
//             }
//         }
//         let sign = value.is_negative();
//         value <<= P;
//         if sign {
//             value |= SIGN_BIT;
//         }
//         FixedPoint(value)
//     }
// }

impl<const P: u8> From<FixedPoint<P>> for f64 {
    fn from(value: FixedPoint<P>) -> f64 {
        value.0 as f64 / 2_i64.pow(P as u32) as f64
    }
}

impl<const P: u8> Add for FixedPoint<P> {
    type Output = Self;

    fn add(self, rhs: Self) -> Self::Output {
        Self(self.0 + rhs.0)
    }
}

impl<const P: u8> Add<isize> for FixedPoint<P> {
    type Output = Self;

    fn add(self, mut rhs: isize) -> Self::Output {
        let sign = rhs.is_negative();
        rhs <<= P;
        if sign {
            rhs |= SIGN_BIT;
        }
        Self(self.0 + rhs)
    }
}

impl<const P: u8> Add<FixedPoint<P>> for isize {
    type Output = FixedPoint<P>;

    fn add(self, rhs: FixedPoint<P>) -> Self::Output {
        rhs + self
    }
}

impl<const P: u8> Sub for FixedPoint<P> {
    type Output = Self;

    fn sub(self, rhs: Self) -> Self::Output {
        Self(self.0 - rhs.0)
    }
}

impl<const P: u8> Sub<isize> for FixedPoint<P> {
    type Output = Self;

    fn sub(self, mut rhs: isize) -> Self::Output {
        let sign = rhs.is_negative();
        rhs <<= P;
        if sign {
            rhs |= SIGN_BIT;
        }
        Self(self.0 - rhs)
    }
}

impl<const P: u8> Sub<FixedPoint<P>> for isize {
    type Output = FixedPoint<P>;

    fn sub(self, rhs: FixedPoint<P>) -> Self::Output {
        rhs - self
    }
}

impl<const P: u8> Neg for FixedPoint<P> {
    type Output = Self;

    fn neg(self) -> Self::Output {
        Self(-self.0)
    }
}

impl<const P: u8> Mul for FixedPoint<P> {
    type Output = Self;

    fn mul(self, rhs: Self) -> Self::Output {
        Self((self.0 * rhs.0) >> P)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn constants_check() {
        // assert that the sign  (1000...000)
        // xor with the exp mask (0111...000)
        // and the mantissa mask (0000...111)
        // should be all ones
        assert_eq!(
            (1 << (doubles::MANTISSA_DIGITS + doubles::EXP_DIGITS))
                ^ doubles::MANTISSA_MASK
                ^ doubles::EXP_MASK,
            u64::MAX
        );
    }

    // #[test]
    // fn print_info() {
    //     println!("isize: {}", isize::BITS);
    //     println!("MANTISSA_DIGITS: {}", doubles::MANTISSA_DIGITS);
    //     println!("EXP_DIGITS: {}", doubles::EXP_DIGITS);
    //     println!("EXP_BIAS: {}", doubles::EXP_BIAS);
    //     print_binary(doubles::EXP_MASK);
    //     print_binary(doubles::MANTISSA_MASK);
    //     panic!();
    // }

    #[test]
    fn init_zero() {
        let a = FixedPoint::<3>::zero();
        assert_eq!(a.0, 0);
    }

    #[test]
    fn from_int() {
        let a: FixedPoint<3> = 1.into();
        assert_eq!(a.0, 0b1000);
        let b = FixedPoint::<3>::from(1);
        assert_eq!(b.0, 0b1000);
    }

    #[test]
    fn from_negative_int() {
        let a: FixedPoint<3> = (-1).into();
        assert_eq!(a.0, -0b1000);
        let b = FixedPoint::<3>::from(-1);
        assert_eq!(b.0, -0b1000);
    }

    #[test]
    fn into_int() {
        let a = FixedPoint::<3>(0b1000);
        assert_eq!(a.into_int(), 1);
    }

    #[test]
    fn into_f64() {
        let a = FixedPoint::<3>(0b1110);
        assert_eq!(f64::from(a), 1.75);
        let b: f64 = a.into();
        assert_eq!(b, 1.75);
    }

    #[test]
    fn into_f64_negative() {
        let a = FixedPoint::<3>(-0b1110);
        assert_eq!(f64::from(a), -1.75);
        let b: f64 = a.into();
        assert_eq!(b, -1.75);
    }

    #[test]
    fn deleteme() {
        let a = FixedPoint::<50>::from_f64(-364.83040918300094);
        assert_eq!(f64::from(a), -364.83040918300094);
    }

    #[test]
    fn from_f64_zero() {
        let a: FixedPoint<3> = FixedPoint::from_f64(0.0);
        assert_eq!(a.0, 0);
    }

    #[test]
    fn from_f64() {
        let a: FixedPoint<3> = FixedPoint::from_f64(1.75);
        assert_eq!(a.0, 0b1110);
    }

    #[test]
    fn from_f64_negative() {
        let a: FixedPoint<3> = FixedPoint::from_f64(-1.75);
        assert_eq!(a.0, -0b1110);
    }

    #[test]
    #[should_panic]
    fn basic_failure_check() {
        let a = FixedPoint::<2>(0b1000);
        assert_eq!(a.into_int(), 1);
    }

    #[test]
    #[should_panic]
    fn from_int_p_too_large() {
        let a = FixedPoint::<64>(0b1000);
        assert_eq!(a.into_int(), 1);
    }

    #[test]
    #[should_panic]
    fn init_zero_p_too_large() {
        let a = FixedPoint::<64>::zero();
        assert_eq!(a.0, 0);
    }

    #[test]
    #[should_panic]
    fn from_f64_p_too_large() {
        let a = FixedPoint::<64>::from_f64(1.0);
        assert_eq!(a.into_int(), 1);
    }

    #[test]
    #[should_panic]
    fn from_int_too_large() {
        let a = FixedPoint::<3>::from(isize::MAX);
        assert_eq!(a.into_int(), 1);
    }

    #[test]
    #[should_panic]
    fn from_int_too_negative() {
        let a = FixedPoint::<3>::from(isize::MIN);
        assert_eq!(a.into_int(), 1);
    }

    #[test]
    fn add() {
        let a = FixedPoint::<3>(0b110);
        let b = FixedPoint::<3>(0b010);
        let c = a + b;
        assert_eq!(c.0, 0b1000);
    }

    #[test]
    fn add_crosses_sign() {
        let a = FixedPoint::<3>(0b100);
        let b = FixedPoint::<3>(-0b110);
        let c = a + b;
        assert_eq!(c.0, -0b10);
    }

    #[test]
    fn sub() {
        let a = FixedPoint::<3>(0b1000);
        let b = FixedPoint::<3>(0b100);
        let c = a - b;
        assert_eq!(c.0, 0b100);
    }

    #[test]
    fn neg() {
        let a = FixedPoint::<3>(0b1000);
        let b = -a;
        assert_eq!(b.0, -0b1000);
    }

    #[test]
    fn add_negative_rhs() {
        let a = FixedPoint::<3>(0b1000);
        let b = -FixedPoint::<3>(0b100);
        let c = a + b;
        assert_eq!(c.0, 0b100);
    }

    #[test]
    fn mul() {
        let a = FixedPoint::<3>(0b100);
        let b = FixedPoint::<3>(0b100);
        let c = a * b;
        assert_eq!(c.0, 0b010);
    }

    #[test]
    fn negative_mul() {
        let a = FixedPoint::<3>(0b100);
        let b = -FixedPoint::<3>(0b100);
        let c = a * b;
        assert_eq!(c.0, -0b010);
    }
}

#[cfg(test)]
mod ptest {
    use super::*;
    use comptime::comptime;
    use proptest::prelude::*;
    use seq_macro::seq;

    seq!(P in 0..=52 {
        const TOLERANCE_~P: f64 = comptime!{2.0_f64.powi(-(P as i32))};
        const FAIL_TOLERANCE_~P: f64 = comptime!{2.0_f64.powi(-(P as i32 + 1))};
        const RANGE_~P: f64 = comptime!{2.0_f64.powi(53 - P as i32)};
        const HALF_RANGE_~P: f64 = comptime!{2.0_f64.powi(52 - P as i32)};

        proptest! {
        #[test]
            fn f64_conversions_~P(x in -RANGE_~P..RANGE_~P) {
                let a = FixedPoint::<P>::from_f64(x);
                assert!((f64::from(a) - x).abs() <= TOLERANCE_~P);
            }

            #[test]
            #[should_panic]
            fn f64_conversions_fail_~P(x in -RANGE_~P..RANGE_~P) {
                let a = FixedPoint::<P>::from_f64(x);
                assert!((f64::from(a) - x).abs() <= FAIL_TOLERANCE_~P);
            }

            #[test]
            fn f64_additions_test_~P(x in -HALF_RANGE_~P..HALF_RANGE_~P, y in -HALF_RANGE_~P..HALF_RANGE_~P) {
                let a = FixedPoint::<P>::from_f64(x);
                let b = FixedPoint::<P>::from_f64(y);
                let c = a + b;
                assert!((f64::from(c) - (x + y)).abs() <= 2.0*TOLERANCE_~P);
            }

            #[test]
            fn f64_subtractions_test_~P(x in -HALF_RANGE_~P..HALF_RANGE_~P, y in -HALF_RANGE_~P..HALF_RANGE_~P) {
                let a = FixedPoint::<P>::from_f64(x);
                let b = FixedPoint::<P>::from_f64(y);
                let c = a - b;
                assert!((f64::from(c) - (x - y)).abs() <= 2.0*TOLERANCE_~P);
            }

        }
    });

    seq!(P in 0..=63u8 {
        const INT_RANGE_~P: i64 = i64::MIN >> P;

        proptest! {
            #[test]
            fn int_in_range_~P(x in INT_RANGE_~P..(!INT_RANGE_~P)) {
                let a = FixedPoint::<P>::from(x);
                let b: FixedPoint<P> = x.into();
            }
        }
    });

    seq!(P in 0..=62u8 {
        const INT_EXTREME_~P: i64 = i64::MIN >> P;
        #[test]
        fn int_too_high_fails_~P() {
            let a = FixedPoint::<P>::from(INT_EXTREME_~P);
        }

        #[test]
        fn int_too_low_fails_~P() {
            let a = FixedPoint::<P>::from(!INT_EXTREME_~P);
        }
    });
}
