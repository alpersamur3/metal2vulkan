//! The GPU's decode of ASTC blocks into a decoded twin's texels: the module
//! [`astc_decode_module`](super::astc_decode_module) serializes, the same decode as the CPU's
//! ([`AstcDecode::decode`]) in integer operations alone.
//!
//! One invocation per texel decodes its block's header, its partition's endpoints and its own
//! weights, straight-line: each reading of the integer sequence encoding (a colour value or a
//! weight) computes the plain, trit and quint forms of its value and selects the range's, and
//! every error condition joins one flag, which turns the texel magenta. The tables of the trits,
//! the quints and the unquantizations are the CPU decode's own functions, tabulated as the module
//! is built. An HDR format's module also unpacks the HDR colour endpoint modes, all four
//! unpackings computed and the mode's selected, and maps each channel to its FP16 value.

use super::{
    quints, trits, unquantize_color, unquantize_weight, AstcDecode, AstcTexels, Range, HALF_ONE,
    RANGES, WEIGHT_RANGES,
};
use crate::bind::block_decode::{table, BlockDecodeModule, BLOCKS, SHAPE, TEXELS};
use crate::bind::ycbcr_convert::{scalar, words_buffer};
use crate::lower::arith::exact_division::div_exact;
use crate::lower::arith::legal::{Form, Held};
use crate::lower::arith::srgb::SRGB_DECODE_STEPS;
use crate::spirv::builder::constructs::{ArmEnd, ArmFn};
use crate::spirv::builder::{FunctionBuilder, ModuleBuilder};
use crate::spirv::error::BResult;
use crate::spirv::ids::{GlobalId, TypeId, Value};
use crate::spirv::ir::{
    BinOp, ConstBits, Constant, ConvOp, ExecMode, ExecModel, GlslOp, IoDir, MemAccess, Type,
    TypedModule, UnOp,
};

/// The module's tables (the module docs), as `Private` arrays.
#[derive(Clone, Copy)]
struct Tables {
    /// The five trits of each 8-bit trit block, 2 bits each.
    trits: GlobalId,
    /// The three quints of each 7-bit quint block, 3 bits each.
    quints: GlobalId,
    /// Each range of [`RANGES`]: its bits, and its base at bit 8.
    ranges: GlobalId,
    /// Each range of [`WEIGHT_RANGES`], likewise.
    weight_ranges: GlobalId,
    /// The first entry of each range of [`RANGES`] in `colors`.
    color_at: GlobalId,
    /// Every colour value of every range of 6 levels or more, unquantized (0 elsewhere).
    colors: GlobalId,
    /// The first entry of each range of [`WEIGHT_RANGES`] in `weights`.
    weight_at: GlobalId,
    /// Every weight of every weight range, unquantized.
    weights: GlobalId,
    /// Metal's sRGB decode step of each 8-bit code.
    srgb_steps: GlobalId,
}

impl Tables {
    fn new(b: &mut ModuleBuilder, u32_: TypeId) -> BResult<Tables> {
        let info = |r: &Range| r.bits | r.base << 8;
        let mut color_at = Vec::with_capacity(RANGES.len());
        let mut colors = Vec::new();
        for range in RANGES {
            color_at.push(colors.len() as u32);
            for v in 0..range.levels() {
                colors.push(match range.levels() >= 6 {
                    true => unquantize_color(range, v),
                    false => 0,
                });
            }
        }
        let mut weight_at = Vec::with_capacity(WEIGHT_RANGES.len());
        let mut weights = Vec::new();
        for range in WEIGHT_RANGES {
            weight_at.push(weights.len() as u32);
            weights.extend((0..range.levels()).map(|v| unquantize_weight(range, v)));
        }
        let trit_words: Vec<u32> = (0..256)
            .map(|t| {
                (trits(t).into_iter())
                    .enumerate()
                    .fold(0, |w, (k, v)| w | v << (2 * k))
            })
            .collect();
        let quint_words: Vec<u32> = (0..128)
            .map(|q| {
                (quints(q).into_iter())
                    .enumerate()
                    .fold(0, |w, (k, v)| w | v << (3 * k))
            })
            .collect();
        let ranges: Vec<u32> = RANGES.iter().map(info).collect();
        let weight_ranges: Vec<u32> = WEIGHT_RANGES.iter().map(info).collect();
        let steps: Vec<u32> = SRGB_DECODE_STEPS.iter().map(|&s| u32::from(s)).collect();
        Ok(Tables {
            trits: table(b, u32_, &trit_words)?,
            quints: table(b, u32_, &quint_words)?,
            ranges: table(b, u32_, &ranges)?,
            weight_ranges: table(b, u32_, &weight_ranges)?,
            color_at: table(b, u32_, &color_at)?,
            colors: table(b, u32_, &colors)?,
            weight_at: table(b, u32_, &weight_at)?,
            weights: table(b, u32_, &weights)?,
            srgb_steps: table(b, u32_, &steps)?,
        })
    }
}

/// A 128-bit stream of four words, its lowest bit first.
type Stream = [Value; 4];

/// A range of the integer sequence encoding, read from a table at decode time: its low bits and
/// its base (1, 3 or 5).
#[derive(Clone, Copy)]
struct Ise {
    bits: Value,
    base: Value,
}

/// The decode of one texel: the function it builds in, and the values every step reads.
struct K<'a, 'b> {
    f: &'a mut FunctionBuilder<'b>,
    u32_: TypeId,
    tables: Tables,
}

impl K<'_, '_> {
    fn k(&mut self, v: u32) -> BResult<Value> {
        scalar(self.f, self.u32_, u64::from(v))
    }

    /// The `bool` constant `v`.
    fn boolean(&mut self, v: bool) -> BResult<Value> {
        let ty = self.f.module().ty(Type::Bool)?;
        let c = self.f.module().constant(Constant::Scalar {
            ty,
            bits: ConstBits(u64::from(v)),
        })?;
        self.f.module().const_value(c)
    }

    /// Whether `v` is one of `values`.
    fn one_of(&mut self, v: Value, values: &[u32]) -> BResult<Value> {
        let mut any = self.boolean(false)?;
        for &value in values {
            let is = self.eqk(v, value)?;
            any = self.lor(any, is)?;
        }
        Ok(any)
    }

    fn op(&mut self, op: BinOp, a: Value, b: Value) -> BResult<Value> {
        self.f.binary(op, a, b)
    }

    fn opk(&mut self, op: BinOp, a: Value, b: u32) -> BResult<Value> {
        let b = self.k(b)?;
        self.f.binary(op, a, b)
    }

    fn add(&mut self, a: Value, b: Value) -> BResult<Value> {
        self.op(BinOp::IAdd, a, b)
    }

    fn addk(&mut self, a: Value, b: u32) -> BResult<Value> {
        self.opk(BinOp::IAdd, a, b)
    }

    fn sub(&mut self, a: Value, b: Value) -> BResult<Value> {
        self.op(BinOp::ISub, a, b)
    }

    fn mul(&mut self, a: Value, b: Value) -> BResult<Value> {
        self.op(BinOp::IMul, a, b)
    }

    fn mulk(&mut self, a: Value, b: u32) -> BResult<Value> {
        self.opk(BinOp::IMul, a, b)
    }

    fn and(&mut self, a: Value, b: Value) -> BResult<Value> {
        self.op(BinOp::BitwiseAnd, a, b)
    }

    fn andk(&mut self, a: Value, b: u32) -> BResult<Value> {
        self.opk(BinOp::BitwiseAnd, a, b)
    }

    fn or(&mut self, a: Value, b: Value) -> BResult<Value> {
        self.op(BinOp::BitwiseOr, a, b)
    }

    fn shl(&mut self, a: Value, b: Value) -> BResult<Value> {
        self.op(BinOp::ShiftLeftLogical, a, b)
    }

    fn shlk(&mut self, a: Value, b: u32) -> BResult<Value> {
        self.opk(BinOp::ShiftLeftLogical, a, b)
    }

    fn shr(&mut self, a: Value, b: Value) -> BResult<Value> {
        self.op(BinOp::ShiftRightLogical, a, b)
    }

    fn shrk(&mut self, a: Value, b: u32) -> BResult<Value> {
        self.opk(BinOp::ShiftRightLogical, a, b)
    }

    fn sel(&mut self, c: Value, a: Value, b: Value) -> BResult<Value> {
        self.f.select(c, a, b)
    }

    fn eqk(&mut self, a: Value, b: u32) -> BResult<Value> {
        self.opk(BinOp::IEqual, a, b)
    }

    fn lor(&mut self, a: Value, b: Value) -> BResult<Value> {
        self.op(BinOp::LogicalOr, a, b)
    }

    fn land(&mut self, a: Value, b: Value) -> BResult<Value> {
        self.op(BinOp::LogicalAnd, a, b)
    }

    fn not(&mut self, a: Value) -> BResult<Value> {
        self.f.unary(UnOp::LogicalNot, a)
    }

    /// `(v >> at) & (2^n - 1)` for a constant field.
    fn field(&mut self, v: Value, at: u32, n: u32) -> BResult<Value> {
        let shifted = self.shrk(v, at)?;
        self.andk(shifted, (1 << n) - 1)
    }

    /// Entry `index` of the `Private` table `table`.
    fn entry(&mut self, table: GlobalId, index: Value) -> BResult<Value> {
        let table = self.f.module().global_value(table)?;
        let p = self.f.access_chain(table, &[index])?;
        self.f.load(p, MemAccess::default())
    }

    /// Word `i` of `stream`, 0 past its fourth.
    fn word(&mut self, stream: &Stream, i: Value) -> BResult<Value> {
        let mut word = self.k(0)?;
        for (k, &w) in (0..4).zip(stream).rev() {
            let at = self.eqk(i, k)?;
            word = self.sel(at, w, word)?;
        }
        Ok(word)
    }

    /// `2^n - 1` for `n` of 0 to 31.
    fn mask(&mut self, n: Value) -> BResult<Value> {
        let one = self.k(1)?;
        let top = self.shl(one, n)?;
        self.sub(top, one)
    }

    /// The `n` bits (0 to 31) of `stream` from bit `at`, a bit past 127 read as 0.
    fn read(&mut self, stream: &Stream, at: Value, n: Value) -> BResult<Value> {
        let i = self.shrk(at, 5)?;
        let shift = self.andk(at, 31)?;
        let low = self.word(stream, i)?;
        let low = self.shr(low, shift)?;
        let next = self.addk(i, 1)?;
        let high = self.word(stream, next)?;
        let thirty_two = self.k(32)?;
        let up = self.sub(thirty_two, shift)?;
        let up = self.andk(up, 31)?;
        let high = self.shl(high, up)?;
        let aligned = self.eqk(shift, 0)?;
        let zero = self.k(0)?;
        let high = self.sel(aligned, zero, high)?;
        let bits = self.or(low, high)?;
        let mask = self.mask(n)?;
        self.and(bits, mask)
    }

    /// [`read`](Self::read) of a stream whose bits from `end` on are zero.
    fn read_to(&mut self, stream: &Stream, at: Value, n: Value, end: Value) -> BResult<Value> {
        let bits = self.read(stream, at, n)?;
        let past = self.op(BinOp::UGreaterThanEqual, at, end)?;
        let left = self.sub(end, at)?;
        let left = self.op(BinOp::ULessThan, left, n)?;
        let available = self.sub(end, at)?;
        let available = self.sel(left, available, n)?;
        let zero = self.k(0)?;
        let available = self.sel(past, zero, available)?;
        let mask = self.mask(available)?;
        self.and(bits, mask)
    }

    /// [`read_to`](Self::read_to) of a constant width.
    fn read_tok(&mut self, stream: &Stream, at: Value, n: u32, end: Value) -> BResult<Value> {
        let n = self.k(n)?;
        self.read_to(stream, at, n, end)
    }

    /// The range of entry `index` of a range table.
    fn ise(&mut self, ranges: GlobalId, index: Value) -> BResult<Ise> {
        let info = self.entry(ranges, index)?;
        let bits = self.andk(info, 0xff)?;
        let base = self.shrk(info, 8)?;
        Ok(Ise { bits, base })
    }

    /// The bits `count` values of `range` take in the encoding.
    fn ise_bits(&mut self, range: Ise, count: Value) -> BResult<Value> {
        let plain = self.mul(count, range.bits)?;
        let eight = self.mulk(count, 8)?;
        let eight = self.addk(eight, 4)?;
        let trit_bits = self.opk(BinOp::UDiv, eight, 5)?;
        let seven = self.mulk(count, 7)?;
        let seven = self.addk(seven, 2)?;
        let quint_bits = self.opk(BinOp::UDiv, seven, 3)?;
        let zero = self.k(0)?;
        let is_trit = self.eqk(range.base, 3)?;
        let is_quint = self.eqk(range.base, 5)?;
        let extra = self.sel(is_quint, quint_bits, zero)?;
        let extra = self.sel(is_trit, trit_bits, extra)?;
        self.add(plain, extra)
    }

    /// Value `i` of the integer sequence encoding of `range` from bit `start` of `stream`, whose
    /// bits from `end` on are zero: its trit or quint times `2^bits` plus its low bits.
    fn ise_value(
        &mut self,
        stream: &Stream,
        (start, end): (Value, Value),
        range: Ise,
        i: Value,
    ) -> BResult<Value> {
        let n = range.bits;
        // Plain: `bits` bits at `start + i · bits`.
        let at = self.mul(i, n)?;
        let at = self.add(start, at)?;
        let plain = self.read_to(stream, at, n, end)?;
        // Trits: groups of five values in `5 bits + 8` bits.
        let group = self.opk(BinOp::UDiv, i, 5)?;
        let first = self.mulk(group, 5)?;
        let k = self.sub(i, first)?;
        let size = self.mulk(n, 5)?;
        let size = self.addk(size, 8)?;
        let size = self.mul(group, size)?;
        let base = self.add(start, size)?;
        let mut t = self.k(0)?;
        // The trit block's bits: 2 after the first value, 2 after the second, 1, 2, 1.
        for (values, gaps, width, at) in [
            (1, 0, 2, 0),
            (2, 2, 2, 2),
            (3, 4, 1, 4),
            (4, 5, 2, 5),
            (5, 7, 1, 7),
        ] {
            let pos = self.mulk(n, values)?;
            let pos = self.addk(pos, gaps)?;
            let pos = self.add(base, pos)?;
            let piece = self.read_tok(stream, pos, width, end)?;
            let piece = self.shlk(piece, at)?;
            t = self.or(t, piece)?;
        }
        let mut gap = self.k(0)?;
        for (index, before) in [(1, 2), (2, 4), (3, 5), (4, 7)] {
            let here = self.eqk(k, index)?;
            let before = self.k(before)?;
            gap = self.sel(here, before, gap)?;
        }
        let at = self.mul(k, n)?;
        let at = self.add(at, gap)?;
        let at = self.add(base, at)?;
        let low = self.read_to(stream, at, n, end)?;
        let trit_word = self.entry(self.tables.trits, t)?;
        let shift = self.shlk(k, 1)?;
        let trit = self.shr(trit_word, shift)?;
        let trit = self.andk(trit, 3)?;
        let trit = self.shl(trit, n)?;
        let trit_value = self.or(trit, low)?;
        // Quints: groups of three values in `3 bits + 7` bits.
        let group = self.opk(BinOp::UDiv, i, 3)?;
        let first = self.mulk(group, 3)?;
        let k = self.sub(i, first)?;
        let size = self.mulk(n, 3)?;
        let size = self.addk(size, 7)?;
        let size = self.mul(group, size)?;
        let base = self.add(start, size)?;
        let mut q = self.k(0)?;
        // The quint block's bits: 3 after the first value, 2 after the second, 2 after the third.
        for (values, gaps, width, at) in [(1, 0, 3, 0), (2, 3, 2, 3), (3, 5, 2, 5)] {
            let pos = self.mulk(n, values)?;
            let pos = self.addk(pos, gaps)?;
            let pos = self.add(base, pos)?;
            let piece = self.read_tok(stream, pos, width, end)?;
            let piece = self.shlk(piece, at)?;
            q = self.or(q, piece)?;
        }
        let mut gap = self.k(0)?;
        for (index, before) in [(1, 3), (2, 5)] {
            let here = self.eqk(k, index)?;
            let before = self.k(before)?;
            gap = self.sel(here, before, gap)?;
        }
        let at = self.mul(k, n)?;
        let at = self.add(at, gap)?;
        let at = self.add(base, at)?;
        let low = self.read_to(stream, at, n, end)?;
        let quint_word = self.entry(self.tables.quints, q)?;
        let shift = self.mulk(k, 3)?;
        let quint = self.shr(quint_word, shift)?;
        let quint = self.andk(quint, 7)?;
        let quint = self.shl(quint, n)?;
        let quint_value = self.or(quint, low)?;
        let is_trit = self.eqk(range.base, 3)?;
        let is_quint = self.eqk(range.base, 5)?;
        let value = self.sel(is_quint, quint_value, plain)?;
        self.sel(is_trit, trit_value, value)
    }

    /// `v` clamped into 0 to 255, as a signed value.
    fn clamp8(&mut self, v: Value) -> BResult<Value> {
        let zero = self.k(0)?;
        let top = self.k(255)?;
        let low = self.op(BinOp::SLessThan, v, zero)?;
        let v = self.sel(low, zero, v)?;
        let high = self.op(BinOp::SGreaterThan, v, top)?;
        self.sel(high, top, v)
    }

    /// `bit_transfer_signed(a, b)`: the signed 6-bit offset and the base.
    fn bit_transfer(&mut self, a: Value, b: Value) -> BResult<(Value, Value)> {
        let b = self.shrk(b, 1)?;
        let top = self.andk(a, 0x80)?;
        let b = self.or(b, top)?;
        let a = self.shrk(a, 1)?;
        let a = self.andk(a, 0x3f)?;
        let negative = self.andk(a, 0x20)?;
        let negative = self.opk(BinOp::INotEqual, negative, 0)?;
        let lowered = self.opk(BinOp::ISub, a, 0x40)?;
        let a = self.sel(negative, lowered, a)?;
        Ok((a, b))
    }

    /// `blue_contract(r, g, b, a)`, signed.
    fn blue_contract(&mut self, [r, g, b, a]: [Value; 4]) -> BResult<[Value; 4]> {
        let r = self.add(r, b)?;
        let r = self.opk(BinOp::ShiftRightArithmetic, r, 1)?;
        let g = self.add(g, b)?;
        let g = self.opk(BinOp::ShiftRightArithmetic, g, 1)?;
        Ok([r, g, b, a])
    }

    fn clamp4(&mut self, e: [Value; 4]) -> BResult<[Value; 4]> {
        Ok([
            self.clamp8(e[0])?,
            self.clamp8(e[1])?,
            self.clamp8(e[2])?,
            self.clamp8(e[3])?,
        ])
    }

    fn sel4(&mut self, c: Value, a: [Value; 4], b: [Value; 4]) -> BResult<[Value; 4]> {
        Ok([
            self.sel(c, a[0], b[0])?,
            self.sel(c, a[1], b[1])?,
            self.sel(c, a[2], b[2])?,
            self.sel(c, a[3], b[3])?,
        ])
    }

    /// The two endpoints of colour endpoint mode `cem` from its unquantized values `v`
    /// (`super::endpoints`), and whether the mode is an HDR one.
    fn endpoints(&mut self, cem: Value, v: [Value; 8]) -> BResult<([[Value; 4]; 2], Value)> {
        let opaque = self.k(255)?;
        let zero = self.k(0)?;
        // 0: luminance.
        let lum = [[v[0], v[0], v[0], opaque], [v[1], v[1], v[1], opaque]];
        // 1: luminance, base and offset.
        let l0 = self.shrk(v[0], 2)?;
        let high = self.andk(v[1], 0xc0)?;
        let l0 = self.or(l0, high)?;
        let offset = self.andk(v[1], 0x3f)?;
        let l1 = self.add(l0, offset)?;
        let past = self.opk(BinOp::UGreaterThan, l1, 255)?;
        let l1 = self.sel(past, opaque, l1)?;
        let lum_delta = [[l0, l0, l0, opaque], [l1, l1, l1, opaque]];
        // 4: luminance and alpha.
        let lum_alpha = [[v[0], v[0], v[0], v[2]], [v[1], v[1], v[1], v[3]]];
        // 5: luminance and alpha, base and offset.
        let (o1, b0) = self.bit_transfer(v[1], v[0])?;
        let (o3, b2) = self.bit_transfer(v[3], v[2])?;
        let l1 = self.add(b0, o1)?;
        let a1 = self.add(b2, o3)?;
        let lum_alpha_delta = [
            self.clamp4([b0, b0, b0, b2])?,
            self.clamp4([l1, l1, l1, a1])?,
        ];
        // 6 and 10: RGB scaled, and alpha.
        let is_ten = self.eqk(cem, 10)?;
        let a0 = self.sel(is_ten, v[4], opaque)?;
        let a1 = self.sel(is_ten, v[5], opaque)?;
        let mut scaled = [zero; 3];
        for (c, slot) in scaled.iter_mut().enumerate() {
            let product = self.mul(v[c], v[3])?;
            *slot = self.shrk(product, 8)?;
        }
        let rgb_scale = [
            [scaled[0], scaled[1], scaled[2], a0],
            [v[0], v[1], v[2], a1],
        ];
        // 8 and 12: RGB, and alpha.
        let is_twelve = self.eqk(cem, 12)?;
        let a0 = self.sel(is_twelve, v[6], opaque)?;
        let a1 = self.sel(is_twelve, v[7], opaque)?;
        let s1 = self.add(v[1], v[3])?;
        let s1 = self.add(s1, v[5])?;
        let s0 = self.add(v[0], v[2])?;
        let s0 = self.add(s0, v[4])?;
        let forward = self.op(BinOp::UGreaterThanEqual, s1, s0)?;
        let e0 = self.clamp4([v[0], v[2], v[4], a0])?;
        let e1 = self.clamp4([v[1], v[3], v[5], a1])?;
        let c0 = self.blue_contract([v[1], v[3], v[5], a1])?;
        let c0 = self.clamp4(c0)?;
        let c1 = self.blue_contract([v[0], v[2], v[4], a0])?;
        let c1 = self.clamp4(c1)?;
        let rgb = [self.sel4(forward, e0, c0)?, self.sel4(forward, e1, c1)?];
        // 9 and 13: RGB, and alpha, base and offset.
        let is_thirteen = self.eqk(cem, 13)?;
        let (o1, b0) = self.bit_transfer(v[1], v[0])?;
        let (o3, b2) = self.bit_transfer(v[3], v[2])?;
        let (o5, b4) = self.bit_transfer(v[5], v[4])?;
        let (o7, b6) = self.bit_transfer(v[7], v[6])?;
        let o7 = self.sel(is_thirteen, o7, zero)?;
        let b6 = self.sel(is_thirteen, b6, opaque)?;
        let sum = self.add(o1, o3)?;
        let sum = self.add(sum, o5)?;
        let forward = self.op(BinOp::SGreaterThanEqual, sum, zero)?;
        let ends = [
            self.add(b0, o1)?,
            self.add(b2, o3)?,
            self.add(b4, o5)?,
            self.add(b6, o7)?,
        ];
        let e0 = self.clamp4([b0, b2, b4, b6])?;
        let e1 = self.clamp4(ends)?;
        let c0 = self.blue_contract(ends)?;
        let c0 = self.clamp4(c0)?;
        let c1 = self.blue_contract([b0, b2, b4, b6])?;
        let c1 = self.clamp4(c1)?;
        let rgb_delta = [self.sel4(forward, e0, c0)?, self.sel4(forward, e1, c1)?];
        // The mode's.
        let mut out = lum;
        let modes: [(&[u32], _); 6] = [
            (&[1], lum_delta),
            (&[4], lum_alpha),
            (&[5], lum_alpha_delta),
            (&[6, 10], rgb_scale),
            (&[8, 12], rgb),
            (&[9, 13], rgb_delta),
        ];
        for (modes, ends) in modes {
            let here = self.one_of(cem, modes)?;
            out = [
                self.sel4(here, ends[0], out[0])?,
                self.sel4(here, ends[1], out[1])?,
            ];
        }
        let hdr = self.one_of(cem, &[2, 3, 7, 11, 14, 15])?;
        Ok((out, hdr))
    }

    // -----------------------------------------------------------------------------------------
    // HDR endpoints and FP16 values (`super::hdr_endpoints`, `super::unorm16_to_half` and the
    // rest), in integer operations; signed values are two's complement words.

    /// Bit `i` of `v`.
    fn bitk(&mut self, v: Value, i: u32) -> BResult<Value> {
        self.field(v, i, 1)
    }

    /// `x | b << at` where `on`, else `x`.
    fn put(&mut self, x: Value, on: Value, b: Value, at: u32) -> BResult<Value> {
        let placed = self.shlk(b, at)?;
        let zero = self.k(0)?;
        let placed = self.sel(on, placed, zero)?;
        self.or(x, placed)
    }

    /// Whether the one-hot `oh` is one of the modes `modes` names: `oh & modes != 0`.
    fn in_modes(&mut self, oh: Value, modes: u32) -> BResult<Value> {
        let m = self.andk(oh, modes)?;
        self.opk(BinOp::INotEqual, m, 0)
    }

    /// The signed `v` clamped into `low` to `high`.
    fn sclamp(&mut self, v: Value, low: u32, high: u32) -> BResult<Value> {
        let lo = self.k(low)?;
        let hi = self.k(high)?;
        let under = self.op(BinOp::SLessThan, v, lo)?;
        let v = self.sel(under, lo, v)?;
        let over = self.op(BinOp::SGreaterThan, v, hi)?;
        self.sel(over, hi, v)
    }

    /// The signed `v`, at least 0.
    fn smax0(&mut self, v: Value) -> BResult<Value> {
        let zero = self.k(0)?;
        let under = self.op(BinOp::SLessThan, v, zero)?;
        self.sel(under, zero, v)
    }

    /// `a` and `b` swapped where `swap`.
    fn swap(&mut self, swap: Value, a: Value, b: Value) -> BResult<(Value, Value)> {
        Ok((self.sel(swap, b, a)?, self.sel(swap, a, b)?))
    }

    /// Mode 2's luminance (`super::hdr_luminance_large`), each endpoint's three channels.
    fn hdr_luminance_large(&mut self, v0: Value, v1: Value) -> BResult<[[Value; 3]; 2]> {
        let ge = self.op(BinOp::UGreaterThanEqual, v1, v0)?;
        let v0s = self.shlk(v0, 4)?;
        let v1s = self.shlk(v1, 4)?;
        let up = self.addk(v1s, 8)?;
        let down = self.addk(v0s, u32::MAX - 7)?;
        let y0 = self.sel(ge, v0s, up)?;
        let y1 = self.sel(ge, v1s, down)?;
        let y0 = self.shlk(y0, 4)?;
        let y1 = self.shlk(y1, 4)?;
        Ok([[y0; 3], [y1; 3]])
    }

    /// Mode 3's luminance (`super::hdr_luminance_small`).
    fn hdr_luminance_small(&mut self, v0: Value, v1: Value) -> BResult<[[Value; 3]; 2]> {
        let top = self.bitk(v0, 7)?;
        let top = self.opk(BinOp::INotEqual, top, 0)?;
        let low = self.andk(v0, 0x7F)?;
        let hi_a = self.andk(v1, 0xE0)?;
        let hi_a = self.shlk(hi_a, 4)?;
        let lo_a = self.shlk(low, 2)?;
        let y0_a = self.or(hi_a, lo_a)?;
        let d_a = self.andk(v1, 0x1F)?;
        let d_a = self.shlk(d_a, 2)?;
        let hi_b = self.andk(v1, 0xF0)?;
        let hi_b = self.shlk(hi_b, 4)?;
        let lo_b = self.shlk(low, 1)?;
        let y0_b = self.or(hi_b, lo_b)?;
        let d_b = self.andk(v1, 0xF)?;
        let d_b = self.shlk(d_b, 1)?;
        let y0 = self.sel(top, y0_a, y0_b)?;
        let d = self.sel(top, d_a, d_b)?;
        let y1 = self.add(y0, d)?;
        let over = self.opk(BinOp::UGreaterThan, y1, 0xFFF)?;
        let most = self.k(0xFFF)?;
        let y1 = self.sel(over, most, y1)?;
        let y0 = self.shlk(y0, 4)?;
        let y1 = self.shlk(y1, 4)?;
        Ok([[y0; 3], [y1; 3]])
    }

    /// Mode 7's RGB, a base and a scale (`super::hdr_rgb_scale`).
    fn hdr_rgb_scale(&mut self, [v0, v1, v2, v3]: [Value; 4]) -> BResult<[[Value; 3]; 2]> {
        let m0 = self.andk(v0, 0xC0)?;
        let m0 = self.shrk(m0, 6)?;
        let m1 = self.bitk(v1, 7)?;
        let m1 = self.shlk(m1, 2)?;
        let m2 = self.bitk(v2, 7)?;
        let m2 = self.shlk(m2, 3)?;
        let modeval = self.or(m0, m1)?;
        let modeval = self.or(modeval, m2)?;
        let high = self.andk(modeval, 0xC)?;
        let not_c = self.opk(BinOp::INotEqual, high, 0xC)?;
        let all = self.eqk(modeval, 0xF)?;
        let zero = self.k(0)?;
        let low2 = self.andk(modeval, 3)?;
        let top2 = self.shrk(modeval, 2)?;
        let maj_c = self.sel(all, zero, low2)?;
        let majcomp = self.sel(not_c, top2, maj_c)?;
        let four = self.k(4)?;
        let five = self.k(5)?;
        let mode_c = self.sel(all, five, four)?;
        let mode = self.sel(not_c, low2, mode_c)?;
        // Red, green, blue and the scale.
        let mut x = [
            self.andk(v0, 0x3F)?,
            self.andk(v1, 0x1F)?,
            self.andk(v2, 0x1F)?,
            self.andk(v3, 0x1F)?,
        ];
        let bit0 = self.bitk(v1, 6)?;
        let bit1 = self.bitk(v1, 5)?;
        let bit2 = self.bitk(v2, 6)?;
        let bit3 = self.bitk(v2, 5)?;
        let bit4 = self.bitk(v3, 7)?;
        let bit5 = self.bitk(v3, 6)?;
        let bit6 = self.bitk(v3, 5)?;
        let one = self.k(1)?;
        let oh = self.shl(one, mode)?;
        let (red, green, blue, scale) = (0, 1, 2, 3);
        for (i, modes, b, at) in [
            (green, 0x30, bit0, 6),
            (green, 0x3A, bit1, 5),
            (blue, 0x30, bit2, 6),
            (blue, 0x3A, bit3, 5),
            (scale, 0x3D, bit6, 5),
            (scale, 0x2D, bit5, 6),
            (scale, 0x04, bit4, 7),
            (red, 0x3B, bit4, 6),
            (red, 0x04, bit3, 6),
            (red, 0x10, bit5, 7),
            (red, 0x0F, bit2, 7),
            (red, 0x05, bit1, 8),
            (red, 0x0A, bit0, 8),
            (red, 0x05, bit0, 9),
            (red, 0x02, bit6, 9),
            (red, 0x01, bit3, 10),
            (red, 0x02, bit5, 10),
        ] {
            let on = self.in_modes(oh, modes)?;
            x[i] = self.put(x[i], on, b, at)?;
        }
        // Up to 12 bits: modes 0 to 5 shift by 1, 1, 2, 3, 4, 5.
        let first = self.eqk(mode, 0)?;
        let shift = self.sel(first, one, mode)?;
        let red = self.shl(x[red], shift)?;
        let green = self.shl(x[green], shift)?;
        let blue = self.shl(x[blue], shift)?;
        let scale = self.shl(x[scale], shift)?;
        let not_five = self.opk(BinOp::INotEqual, mode, 5)?;
        let green_d = self.sub(red, green)?;
        let blue_d = self.sub(red, blue)?;
        let green = self.sel(not_five, green_d, green)?;
        let blue = self.sel(not_five, blue_d, blue)?;
        let is1 = self.eqk(majcomp, 1)?;
        let is2 = self.eqk(majcomp, 2)?;
        let (red, green) = self.swap(is1, red, green)?;
        let (red, blue) = self.swap(is2, red, blue)?;
        let mut ends = [[zero; 3]; 2];
        for (ch, c) in [red, green, blue].into_iter().enumerate() {
            let high = self.smax0(c)?;
            ends[1][ch] = self.shlk(high, 4)?;
            let low = self.sub(c, scale)?;
            let low = self.smax0(low)?;
            ends[0][ch] = self.shlk(low, 4)?;
        }
        Ok(ends)
    }

    /// Modes 11, 14 and 15's RGB (`super::hdr_rgb`).
    fn hdr_rgb(&mut self, v: [Value; 6]) -> BResult<[[Value; 3]; 2]> {
        let [v0, v1, v2, v3, v4, v5] = v;
        let mut top = [v0; 6];
        for (slot, &x) in top.iter_mut().zip(&v) {
            *slot = self.bitk(x, 7)?;
        }
        let m1 = self.shlk(top[2], 1)?;
        let m2 = self.shlk(top[3], 2)?;
        let modeval = self.or(top[1], m1)?;
        let modeval = self.or(modeval, m2)?;
        let j1 = self.shlk(top[5], 1)?;
        let majcomp = self.or(top[4], j1)?;
        // Direct: the major-channel field 3.
        let mut direct = [[v0; 3]; 2];
        for (end, (r, g, b)) in direct.iter_mut().zip([(v0, v2, v4), (v1, v3, v5)]) {
            end[0] = self.shlk(r, 8)?;
            end[1] = self.shlk(g, 8)?;
            let b = self.andk(b, 0x7F)?;
            end[2] = self.shlk(b, 9)?;
        }
        let a40 = self.andk(v1, 0x40)?;
        let a40 = self.shlk(a40, 2)?;
        // `a`, `b0`, `b1`, `c`, `d0` and `d1`.
        let mut x = [
            self.or(v0, a40)?,
            self.andk(v2, 0x3F)?,
            self.andk(v3, 0x3F)?,
            self.andk(v1, 0x3F)?,
            self.andk(v4, 0x7F)?,
            self.andk(v5, 0x7F)?,
        ];
        // The differences' bits, 7, 6, 7, 6, 5, 6, 5, 6 by mode: 6 for an odd mode, else 5 for
        // modes 4 and 6 and 7 for 0 and 2.
        let odd = self.andk(modeval, 1)?;
        let odd = self.opk(BinOp::INotEqual, odd, 0)?;
        let four = self.andk(modeval, 4)?;
        let four = self.opk(BinOp::INotEqual, four, 0)?;
        let five = self.k(5)?;
        let six = self.k(6)?;
        let seven = self.k(7)?;
        let even_bits = self.sel(four, five, seven)?;
        let dbits = self.sel(odd, six, even_bits)?;
        let bit0 = self.bitk(v2, 6)?;
        let bit1 = self.bitk(v3, 6)?;
        let bit2 = self.bitk(v4, 6)?;
        let bit3 = self.bitk(v5, 6)?;
        let bit4 = self.bitk(v4, 5)?;
        let bit5 = self.bitk(v5, 5)?;
        let one = self.k(1)?;
        let oh = self.shl(one, modeval)?;
        let (a, b0, b1, c, d0, d1) = (0, 1, 2, 3, 4, 5);
        for (i, modes, b, at) in [
            (a, 0xA4, bit0, 9),
            (a, 0x08, bit2, 9),
            (a, 0x50, bit4, 9),
            (a, 0x50, bit5, 10),
            (a, 0xA0, bit1, 10),
            (a, 0xC0, bit2, 11),
            (c, 0x04, bit1, 6),
            (c, 0xE8, bit3, 6),
            (c, 0x20, bit2, 7),
            (b0, 0x5B, bit0, 6),
            (b1, 0x5B, bit1, 6),
            (b0, 0x12, bit2, 7),
            (b1, 0x12, bit3, 7),
            (d0, 0xAF, bit4, 5),
            (d1, 0xAF, bit5, 5),
            (d0, 0x05, bit2, 6),
            (d1, 0x05, bit3, 6),
        ] {
            let on = self.in_modes(oh, modes)?;
            x[i] = self.put(x[i], on, b, at)?;
        }
        // `d0` and `d1` are signed `dbits`-bit values.
        let thirty_two = self.k(32)?;
        let sx = self.sub(thirty_two, dbits)?;
        for d in [d0, d1] {
            let up = self.shl(x[d], sx)?;
            x[d] = self.op(BinOp::ShiftRightArithmetic, up, sx)?;
        }
        // Up to 12 bits.
        let half = self.shrk(modeval, 1)?;
        let shift = self.opk(BinOp::BitwiseXor, half, 3)?;
        for slot in x.iter_mut() {
            *slot = self.shl(*slot, shift)?;
        }
        let g1 = self.sub(x[a], x[b0])?;
        let bl1 = self.sub(x[a], x[b1])?;
        let r0 = self.sub(x[a], x[c])?;
        let g0 = self.sub(g1, x[c])?;
        let g0 = self.sub(g0, x[d0])?;
        let bl0 = self.sub(bl1, x[c])?;
        let bl0 = self.sub(bl0, x[d1])?;
        let mut zero_end = [r0, g0, bl0];
        let mut one_end = [x[a], g1, bl1];
        for x in zero_end.iter_mut().chain(one_end.iter_mut()) {
            *x = self.sclamp(*x, 0, 0xFFF)?;
        }
        let is1 = self.eqk(majcomp, 1)?;
        let is2 = self.eqk(majcomp, 2)?;
        for end in [&mut zero_end, &mut one_end] {
            let (r, g) = self.swap(is1, end[0], end[1])?;
            let (r, b) = self.swap(is2, r, end[2])?;
            *end = [r, g, b];
        }
        let is3 = self.eqk(majcomp, 3)?;
        let mut ends = [zero_end, one_end];
        for (end, direct) in ends.iter_mut().zip(direct) {
            for (x, d) in end.iter_mut().zip(direct) {
                let up = self.shlk(*x, 4)?;
                *x = self.sel(is3, d, up)?;
            }
        }
        Ok(ends)
    }

    /// Mode 15's alpha (`super::hdr_alpha`).
    fn hdr_alpha(&mut self, v6: Value, v7: Value) -> BResult<[Value; 2]> {
        let m0 = self.bitk(v6, 7)?;
        let m1 = self.shrk(v7, 6)?;
        let m1 = self.andk(m1, 2)?;
        let modeval = self.or(m0, m1)?;
        let a6 = self.andk(v6, 0x7F)?;
        let a7 = self.andk(v7, 0x7F)?;
        let direct0 = self.shlk(a6, 5)?;
        let direct1 = self.shlk(a7, 5)?;
        let up = self.addk(modeval, 1)?;
        let moved = self.shl(a7, up)?;
        let moved = self.andk(moved, 0x780)?;
        let base = self.or(a6, moved)?;
        let mask = self.k(0x3F)?;
        let mask = self.shr(mask, modeval)?;
        let bias = self.k(32)?;
        let bias = self.shr(bias, modeval)?;
        let delta = self.and(a7, mask)?;
        let delta = self.op(BinOp::BitwiseXor, delta, bias)?;
        let delta = self.sub(delta, bias)?;
        let four = self.k(4)?;
        let shift = self.sub(four, modeval)?;
        let base = self.shl(base, shift)?;
        let delta = self.shl(delta, shift)?;
        let sum = self.add(base, delta)?;
        let sum = self.sclamp(sum, 0, 0xFFF)?;
        let direct = self.eqk(modeval, 3)?;
        let a0 = self.sel(direct, direct0, base)?;
        let a1 = self.sel(direct, direct1, sum)?;
        Ok([self.shlk(a0, 4)?, self.shlk(a1, 4)?])
    }

    /// The endpoints of an HDR colour endpoint mode `cem` from its unquantized values `v`
    /// (`super::hdr_endpoints`), in the interpolator's 16 bits, and whether its alpha is HDR
    /// (all but mode 14's).
    fn hdr_endpoints(&mut self, cem: Value, v: [Value; 8]) -> BResult<([[Value; 4]; 2], Value)> {
        let large = self.hdr_luminance_large(v[0], v[1])?;
        let small = self.hdr_luminance_small(v[0], v[1])?;
        let scale = self.hdr_rgb_scale([v[0], v[1], v[2], v[3]])?;
        let mut rgb = self.hdr_rgb([v[0], v[1], v[2], v[3], v[4], v[5]])?;
        for (mode, ends) in [(7, scale), (3, small), (2, large)] {
            let here = self.eqk(cem, mode)?;
            for (end, own) in rgb.iter_mut().zip(ends) {
                for (x, o) in end.iter_mut().zip(own) {
                    *x = self.sel(here, o, *x)?;
                }
            }
        }
        let hdr_alpha = self.hdr_alpha(v[6], v[7])?;
        let is14 = self.eqk(cem, 14)?;
        let is15 = self.eqk(cem, 15)?;
        let opaque = self.k(0x7800)?;
        let mut alpha = [opaque; 2];
        for (i, slot) in alpha.iter_mut().enumerate() {
            let ldr = self.mulk(v[6 + i], 257)?;
            let a = self.sel(is14, ldr, opaque)?;
            *slot = self.sel(is15, hdr_alpha[i], a)?;
        }
        let lns_alpha = self.not(is14)?;
        let ends = [0, 1].map(|e| [rgb[e][0], rgb[e][1], rgb[e][2], alpha[e]]);
        Ok((ends, lns_alpha))
    }

    /// `super::unorm16_to_half(c)`.
    fn unorm16_to_half(&mut self, c: Value) -> BResult<Value> {
        let msb = self.f.glsl(GlslOp::FindUMsb, &[c])?;
        let sixteen = self.k(16)?;
        let up = self.sub(sixteen, msb)?;
        // A `c` under 4 shifts by 15 to 17 here, a value the subnormal choice replaces.
        let below = self.shl(c, up)?;
        let below = self.andk(below, 0xFFFF)?;
        let mantissa = self.shrk(below, 6)?;
        let exponent = self.addk(msb, u32::MAX)?;
        let exponent = self.shlk(exponent, 10)?;
        let normal = self.or(exponent, mantissa)?;
        let small = self.opk(BinOp::ULessThan, c, 4)?;
        let subnormal = self.shlk(c, 8)?;
        let h = self.sel(small, subnormal, normal)?;
        let full = self.eqk(c, 0xFFFF)?;
        let one = self.k(u32::from(HALF_ONE))?;
        self.sel(full, one, h)
    }

    /// `super::lns_to_half(c)`.
    fn lns_to_half(&mut self, c: Value) -> BResult<Value> {
        let e = self.shrk(c, 11)?;
        let m = self.andk(c, 0x7FF)?;
        let m3 = self.mulk(m, 3)?;
        let m4 = self.mulk(m, 4)?;
        let m4 = self.addk(m4, u32::MAX - 511)?;
        let m5 = self.mulk(m, 5)?;
        let m5 = self.addk(m5, u32::MAX - 2047)?;
        let under_512 = self.opk(BinOp::ULessThan, m, 512)?;
        let under_1536 = self.opk(BinOp::ULessThan, m, 1536)?;
        let mt = self.sel(under_1536, m4, m5)?;
        let mt = self.sel(under_512, m3, mt)?;
        let e = self.shlk(e, 10)?;
        let mt = self.shrk(mt, 3)?;
        let h = self.or(e, mt)?;
        let over = self.opk(BinOp::UGreaterThan, h, 0x7BFF)?;
        let most = self.k(0x7BFF)?;
        self.sel(over, most, h)
    }

    /// The bits of the `f32` of the FP16 value `h` (`super::half_to_f32`).
    fn half_to_f32_bits(&mut self, h: Value) -> BResult<Value> {
        let sign = self.shrk(h, 15)?;
        let sign = self.shlk(sign, 31)?;
        let e = self.field(h, 10, 5)?;
        let m = self.andk(h, 0x3FF)?;
        let m13 = self.shlk(m, 13)?;
        let e32 = self.addk(e, 112)?;
        let e32 = self.shlk(e32, 23)?;
        let normal = self.or(sign, e32)?;
        let normal = self.or(normal, m13)?;
        let top = self.k(0xFF << 23)?;
        let special = self.or(sign, top)?;
        let special = self.or(special, m13)?;
        // Subnormal: `m · 2^-24`, its leading one at bit `msb` (a zero `m` shifts by 24 here, a
        // value the zero choice replaces).
        let msb = self.f.glsl(GlslOp::FindUMsb, &[m])?;
        let e_sub = self.addk(msb, 103)?;
        let e_sub = self.shlk(e_sub, 23)?;
        let twenty_three = self.k(23)?;
        let up = self.sub(twenty_three, msb)?;
        let m_sub = self.shl(m, up)?;
        let m_sub = self.andk(m_sub, 0x7F_FFFF)?;
        let sub = self.or(sign, e_sub)?;
        let sub = self.or(sub, m_sub)?;
        let no_mantissa = self.eqk(m, 0)?;
        let small = self.sel(no_mantissa, sign, sub)?;
        let is_zero_e = self.eqk(e, 0)?;
        let is_top_e = self.eqk(e, 0x1F)?;
        let bits = self.sel(is_top_e, special, normal)?;
        self.sel(is_zero_e, small, bits)
    }

    /// `super::unorm8_of_half(h)`.
    fn unorm8_of_half(&mut self, h: Value) -> BResult<Value> {
        let negative = self.andk(h, 0x8000)?;
        let negative = self.opk(BinOp::INotEqual, negative, 0)?;
        let zero_h = self.eqk(h, 0)?;
        let none = self.lor(negative, zero_h)?;
        let full = self.opk(BinOp::UGreaterThanEqual, h, u32::from(HALF_ONE))?;
        let e = self.field(h, 10, 5)?;
        let m = self.andk(h, 0x3FF)?;
        let lead = self.opk(BinOp::BitwiseOr, m, 0x400)?;
        let e_zero = self.eqk(e, 0)?;
        let less = self.addk(e, u32::MAX)?;
        // Below 1.0 the exponent is at most 14; a larger one (1.0 or more, or a negative value's)
        // shifts by 15 to 30 here, a value the full or the none choice replaces.
        let zero = self.k(0)?;
        let shift = self.sel(e_zero, zero, less)?;
        let normal = self.shl(lead, shift)?;
        let units = self.sel(e_zero, m, normal)?;
        let scaled = self.mulk(units, 255)?;
        let scaled = self.addk(scaled, 1 << 23)?;
        let code = self.shrk(scaled, 24)?;
        let top = self.k(255)?;
        let code = self.sel(full, top, code)?;
        self.sel(none, zero, code)
    }

    /// The FP16 value nearest `num / den` (`super::half_of_ratio`), `den` an odd constant of 4095
    /// or less.
    fn half_of_ratio(&mut self, num: Value, den: u32) -> BResult<Value> {
        // The least `p` of 1 to 12 with `num · 2^p ≥ den`.
        let mut p = self.k(12)?;
        for i in (1..12).rev() {
            let q = self.shlk(num, i)?;
            let fits = self.opk(BinOp::UGreaterThanEqual, q, den)?;
            let here = self.k(i)?;
            p = self.sel(fits, here, p)?;
        }
        let q = self.shl(num, p)?;
        let t = self.mulk(q, 2048)?;
        let t = self.addk(t, den)?;
        let m = self.opk(BinOp::UDiv, t, 2 * den)?;
        let fifteen = self.k(15)?;
        let e = self.sub(fifteen, p)?;
        let e = self.shlk(e, 10)?;
        let h = self.add(e, m)?;
        let h = self.addk(h, u32::MAX - 1023)?;
        let full = self.opk(BinOp::UGreaterThanEqual, num, den)?;
        let one = self.k(u32::from(HALF_ONE))?;
        let h = self.sel(full, one, h)?;
        let nothing = self.eqk(num, 0)?;
        let zero = self.k(0)?;
        self.sel(nothing, zero, h)
    }

    /// `hash52(p)`.
    fn hash52(&mut self, p: Value) -> BResult<Value> {
        let xor = |k: &mut Self, a: Value, b: Value| k.op(BinOp::BitwiseXor, a, b);
        let t = self.shrk(p, 15)?;
        let p = xor(self, p, t)?;
        let t = self.shlk(p, 17)?;
        let p = self.sub(p, t)?;
        let t = self.shlk(p, 7)?;
        let p = self.add(p, t)?;
        let t = self.shlk(p, 4)?;
        let p = self.add(p, t)?;
        let t = self.shrk(p, 5)?;
        let p = xor(self, p, t)?;
        let t = self.shlk(p, 16)?;
        let p = self.add(p, t)?;
        let t = self.shrk(p, 7)?;
        let p = xor(self, p, t)?;
        let t = self.shrk(p, 3)?;
        let p = xor(self, p, t)?;
        let t = self.shlk(p, 6)?;
        let p = xor(self, p, t)?;
        let t = self.shrk(p, 17)?;
        xor(self, p, t)
    }

    /// The partition of texel `(x, y)` of a block of `texels` texels, `count` partitions (2 to
    /// 4) and partition index `seed` (`super::Partitions`).
    fn partition(
        &mut self,
        seed: Value,
        count: Value,
        texels: u32,
        [x, y]: [Value; 2],
    ) -> BResult<Value> {
        let seed = {
            let less = self.addk(count, u32::MAX)?;
            let shifted = self.shlk(less, 10)?;
            self.add(seed, shifted)?
        };
        let rnum = self.hash52(seed)?;
        let mut s = [rnum; 8];
        for (i, slot) in s.iter_mut().enumerate() {
            *slot = self.field(rnum, 4 * i as u32, 4)?;
        }
        let four = self.k(4)?;
        let five = self.k(5)?;
        let six = self.k(6)?;
        let three = self.eqk(count, 3)?;
        let odd = self.andk(seed, 1)?;
        let odd = self.opk(BinOp::INotEqual, odd, 0)?;
        let two = self.andk(seed, 2)?;
        let two = self.opk(BinOp::INotEqual, two, 0)?;
        let by_two = self.sel(two, four, five)?;
        let by_three = self.sel(three, six, five)?;
        let sh1 = self.sel(odd, by_two, by_three)?;
        let sh2 = self.sel(odd, by_three, by_two)?;
        for (i, slot) in s.iter_mut().enumerate() {
            let square = self.mul(*slot, *slot)?;
            let shift = if i % 2 == 0 { sh1 } else { sh2 };
            *slot = self.shr(square, shift)?;
        }
        let (x, y) = match texels < 31 {
            true => (self.shlk(x, 1)?, self.shlk(y, 1)?),
            false => (x, y),
        };
        let mut lines = [rnum; 4];
        for (line, (slot, shift)) in lines.iter_mut().zip([14, 10, 6, 2]).enumerate() {
            let sx = self.mul(s[2 * line], x)?;
            let sy = self.mul(s[2 * line + 1], y)?;
            let sum = self.add(sx, sy)?;
            let r = self.shrk(rnum, shift)?;
            let sum = self.add(sum, r)?;
            *slot = self.andk(sum, 0x3f)?;
        }
        let zero = self.k(0)?;
        let under_three = self.opk(BinOp::ULessThan, count, 3)?;
        let under_four = self.opk(BinOp::ULessThan, count, 4)?;
        let [a, b, c, d] = lines;
        let c = self.sel(under_three, zero, c)?;
        let d = self.sel(under_four, zero, d)?;
        let ge = |k: &mut Self, p: Value, q: Value| k.op(BinOp::UGreaterThanEqual, p, q);
        let ab = ge(self, a, b)?;
        let ac = ge(self, a, c)?;
        let ad = ge(self, a, d)?;
        let bc = ge(self, b, c)?;
        let bd = ge(self, b, d)?;
        let cd = ge(self, c, d)?;
        let first = self.land(ab, ac)?;
        let first = self.land(first, ad)?;
        let second = self.land(bc, bd)?;
        let one = self.k(1)?;
        let two = self.k(2)?;
        let three = self.k(3)?;
        let p = self.sel(cd, two, three)?;
        let p = self.sel(second, one, p)?;
        self.sel(first, zero, p)
    }

    /// Entry `v` of the unquantization table of `range` (an index into its range table) whose
    /// ranges start at `at` in `values`.
    fn unquantize(
        &mut self,
        (at, values): (GlobalId, GlobalId),
        range: Value,
        v: Value,
    ) -> BResult<Value> {
        let first = self.entry(at, range)?;
        let index = self.add(first, v)?;
        self.entry(values, index)
    }
}

/// The texel's channels, each its 16-bit interpolation result `C` (an HDR format's FP16 value),
/// and whether the texel is an error one: the decode of the block `words` at texel `(s, t)` of a
/// block of `decode`'s footprint (`super::decode_block` and `super::void_extent`).
fn texel(
    k: &mut K<'_, '_>,
    decode: AstcDecode,
    words: Stream,
    [s, t]: [Value; 2],
) -> BResult<([Value; 4], Value)> {
    let (bw, bh) = (decode.width, decode.height);
    let zero = k.k(0)?;
    let no = k.boolean(false)?;
    let w0 = words[0];
    let mode = k.andk(w0, 0x7ff)?;
    let bit = |k: &mut K<'_, '_>, i: u32| k.field(mode, i, 1);
    let bits = |k: &mut K<'_, '_>, i: u32, n: u32| k.field(mode, i, n);
    // The block mode (`super::block_mode`): the first form where its low two bits are set.
    let low2 = k.andk(mode, 3)?;
    let first_form = k.opk(BinOp::INotEqual, low2, 0)?;
    let b4 = bit(k, 4)?;
    let b8 = bit(k, 8)?;
    let b9 = bit(k, 9)?;
    let b10 = bit(k, 10)?;
    let a = bits(k, 5, 2)?;
    let b = bits(k, 7, 2)?;
    let low2_up = k.shlk(low2, 1)?;
    let range_first = k.or(b4, low2_up)?;
    let a2 = k.addk(a, 2)?;
    let b4w = k.addk(b, 4)?;
    let b8w = k.addk(b, 8)?;
    let b_low = k.andk(b, 1)?;
    let b6 = k.addk(b_low, 6)?;
    let b2 = k.addk(b_low, 2)?;
    let layout = bits(k, 2, 2)?;
    let is0 = k.eqk(layout, 0)?;
    let is1 = k.eqk(layout, 1)?;
    let is2 = k.eqk(layout, 2)?;
    let high8 = k.opk(BinOp::INotEqual, b8, 0)?;
    let gw3 = k.sel(high8, b2, a2)?;
    let gh3 = k.sel(high8, a2, b6)?;
    let gw = k.sel(is2, a2, gw3)?;
    let gh = k.sel(is2, b8w, gh3)?;
    let gw = k.sel(is1, b8w, gw)?;
    let gh = k.sel(is1, a2, gh)?;
    let gw_first = k.sel(is0, b4w, gw)?;
    let gh_first = k.sel(is0, a2, gh)?;
    // The second form.
    let layout_up = k.shlk(layout, 1)?;
    let range_second = k.or(b4, layout_up)?;
    let twelve = k.k(12)?;
    let a6 = k.addk(a, 6)?;
    let c9 = bits(k, 9, 2)?;
    let c6 = k.addk(c9, 6)?;
    let six = k.k(6)?;
    let ten = k.k(10)?;
    let c0 = k.eqk(b, 0)?;
    let c1 = k.eqk(b, 1)?;
    let c2 = k.eqk(b, 2)?;
    let a0 = k.eqk(a, 0)?;
    let gw3 = k.sel(a0, six, ten)?;
    let gh3 = k.sel(a0, ten, six)?;
    let gw = k.sel(c2, a6, gw3)?;
    let gh = k.sel(c2, c6, gh3)?;
    let gw = k.sel(c1, a2, gw)?;
    let gh = k.sel(c1, twelve, gh)?;
    let gw_second = k.sel(c0, twelve, gw)?;
    let gh_second = k.sel(c0, a2, gh)?;
    let plain_high = k.sel(c2, zero, b9)?;
    let plain_dual = k.sel(c2, zero, b10)?;
    let low4 = k.andk(mode, 0xf)?;
    let reserved_low = k.eqk(low4, 0)?;
    let a_past = k.opk(BinOp::UGreaterThan, a, 1)?;
    let c3 = k.eqk(b, 3)?;
    let reserved_c3 = k.land(c3, a_past)?;
    let reserved_second = k.lor(reserved_low, reserved_c3)?;
    let range_bits = k.sel(first_form, range_first, range_second)?;
    let high = k.sel(first_form, b9, plain_high)?;
    let dual = k.sel(first_form, b10, plain_dual)?;
    let gw = k.sel(first_form, gw_first, gw_second)?;
    let gh = k.sel(first_form, gh_first, gh_second)?;
    let reserved = k.sel(first_form, no, reserved_second)?;
    let few_range = k.opk(BinOp::ULessThan, range_bits, 2)?;
    let mut bad = k.lor(reserved, few_range)?;
    let high_at = k.mulk(high, 6)?;
    let weight_range = k.addk(range_bits, u32::MAX - 1)?;
    let weight_range = k.add(weight_range, high_at)?;
    let weight_range = k.andk(weight_range, 15)?;
    let eleven = k.k(11)?;
    let past_table = k.op(BinOp::UGreaterThan, weight_range, eleven)?;
    let weight_range = k.sel(past_table, zero, weight_range)?;
    let wise = k.ise(k.tables.weight_ranges, weight_range)?;
    let dual = k.opk(BinOp::INotEqual, dual, 0)?;
    let one = k.k(1)?;
    let two = k.k(2)?;
    let planes = k.sel(dual, two, one)?;
    // The grid fits the block, holds at most 64 weights in 24 to 96 bits.
    let wide = k.opk(BinOp::UGreaterThan, gw, bw)?;
    let tall = k.opk(BinOp::UGreaterThan, gh, bh)?;
    bad = k.lor(bad, wide)?;
    bad = k.lor(bad, tall)?;
    let grid = k.mul(gw, gh)?;
    let weight_count = k.mul(grid, planes)?;
    let many = k.opk(BinOp::UGreaterThan, weight_count, 64)?;
    bad = k.lor(bad, many)?;
    let weight_bits = k.ise_bits(wise, weight_count)?;
    let few = k.opk(BinOp::ULessThan, weight_bits, 24)?;
    let lots = k.opk(BinOp::UGreaterThan, weight_bits, 96)?;
    bad = k.lor(bad, few)?;
    bad = k.lor(bad, lots)?;
    // Partitions and colour endpoint modes.
    let count = k.field(w0, 11, 2)?;
    let count = k.addk(count, 1)?;
    let four_parts = k.eqk(count, 4)?;
    let dual_four = k.land(dual, four_parts)?;
    bad = k.lor(bad, dual_four)?;
    let single = k.eqk(count, 1)?;
    let below = k.k(128)?;
    let below = k.sub(below, weight_bits)?;
    let cem_single = k.field(w0, 13, 4)?;
    let seed = k.field(w0, 13, 10)?;
    let low = k.field(w0, 23, 6)?;
    let low_class = k.andk(low, 3)?;
    let shared = k.eqk(low_class, 0)?;
    let three_count = k.mulk(count, 3)?;
    let extra_config = k.addk(three_count, u32::MAX - 3)?;
    let extra_multi = k.sel(shared, zero, extra_config)?;
    let extra = k.sel(single, zero, extra_multi)?;
    let past_weights = k.op(BinOp::UGreaterThan, extra, below)?;
    bad = k.lor(bad, past_weights)?;
    let config_at = k.sub(below, extra)?;
    let config_high = k.read(&words, config_at, extra)?;
    let config_high = k.shlk(config_high, 6)?;
    let config = k.or(low, config_high)?;
    let class_base = k.andk(config, 3)?;
    let class_base = k.addk(class_base, u32::MAX)?;
    let shared_cem = k.shrk(low, 2)?;
    let mut cems = [zero; 4];
    for (i, cem) in (0u32..).zip(cems.iter_mut()) {
        let class_shift = k.k(2 + i)?;
        let class = k.shr(config, class_shift)?;
        let class = k.andk(class, 1)?;
        let class = k.add(class_base, class)?;
        let class = k.shlk(class, 2)?;
        let m_shift = k.addk(count, 2 + 2 * i)?;
        let m = k.shr(config, m_shift)?;
        let m = k.andk(m, 3)?;
        let own = k.or(class, m)?;
        let multi = k.sel(shared, shared_cem, own)?;
        *cem = match i {
            0 => k.sel(single, cem_single, multi)?,
            _ => multi,
        };
    }
    let color_start = {
        let seventeen = k.k(17)?;
        let twenty_nine = k.k(29)?;
        k.sel(single, seventeen, twenty_nine)?
    };
    let plane_bits = k.sel(dual, two, zero)?;
    let reserved_below = k.add(extra, plane_bits)?;
    let short = k.op(BinOp::UGreaterThan, reserved_below, below)?;
    bad = k.lor(bad, short)?;
    let color_end = k.sub(below, reserved_below)?;
    let two_bits = k.k(2)?;
    let ccs = k.read(&words, color_end, two_bits)?;
    let mut value_count = zero;
    let mut counts = [zero; 4];
    for (i, cem) in cems.into_iter().enumerate() {
        let class = k.shrk(cem, 2)?;
        let n = k.addk(class, 1)?;
        let n = k.shlk(n, 1)?;
        let inside = k.opk(BinOp::UGreaterThan, count, i as u32)?;
        counts[i] = k.sel(inside, n, zero)?;
        value_count = k.add(value_count, counts[i])?;
    }
    let too_many = k.opk(BinOp::UGreaterThan, value_count, 18)?;
    bad = k.lor(bad, too_many)?;
    let backwards = k.op(BinOp::UGreaterThan, color_start, color_end)?;
    bad = k.lor(bad, backwards)?;
    let color_bits = k.sub(color_end, color_start)?;
    // The largest colour range of 6 levels or more whose encoding fits.
    let mut color_range = zero;
    let mut color_used = zero;
    for (index, range) in RANGES.iter().enumerate().filter(|(_, r)| r.levels() >= 6) {
        let info = Ise {
            bits: k.k(range.bits)?,
            base: k.k(range.base)?,
        };
        let used = k.ise_bits(info, value_count)?;
        let fits = k.op(BinOp::ULessThanEqual, used, color_bits)?;
        let index = k.k(index as u32)?;
        color_range = k.sel(fits, index, color_range)?;
        color_used = k.sel(fits, used, color_used)?;
    }
    let no_range = k.eqk(color_range, 0)?;
    bad = k.lor(bad, no_range)?;
    let cise = k.ise(k.tables.ranges, color_range)?;
    let color_stream_end = k.add(color_start, color_used)?;
    // The texel's partition, its mode and its values.
    let multi = k.not(single)?;
    let texels = bw * bh;
    let p = k.partition(seed, count, texels, [s, t])?;
    let p = k.sel(multi, p, zero)?;
    let mut cem = cems[0];
    let mut first_value = zero;
    let mut before = zero;
    for i in 1..4u32 {
        before = k.add(before, counts[i as usize - 1])?;
        let here = k.eqk(p, i)?;
        cem = k.sel(here, cems[i as usize], cem)?;
        first_value = k.sel(here, before, first_value)?;
    }
    let mut v = [zero; 8];
    for (i, slot) in (0u32..).zip(v.iter_mut()) {
        let index = k.addk(first_value, i)?;
        let raw = k.ise_value(&words, (color_start, color_stream_end), cise, index)?;
        let tables = (k.tables.color_at, k.tables.colors);
        *slot = k.unquantize(tables, color_range, raw)?;
    }
    let ([e0, e1], hdr) = k.endpoints(cem, v)?;
    // An HDR format's HDR modes unpack their own endpoints; an LDR or sRGB format's read magenta.
    let hdr_ends = match decode.hdr {
        true => Some(k.hdr_endpoints(cem, v)?),
        false => None,
    };
    let partition_bad = match decode.hdr {
        true => bad,
        false => k.lor(bad, hdr)?,
    };
    // The weights, from bit 127 down: the stream of the block's bits reversed.
    let mut reversed = words;
    for (slot, &w) in reversed.iter_mut().zip(words.iter().rev()) {
        *slot = k.f.unary(UnOp::BitReverse, w)?;
    }
    let ds = (1024 + bw / 2) / (bw - 1);
    let dt = (1024 + bh / 2) / (bh - 1);
    let gw_less = k.addk(gw, u32::MAX)?;
    let gh_less = k.addk(gh, u32::MAX)?;
    let gs = k.mulk(s, ds)?;
    let gs = k.mul(gs, gw_less)?;
    let gs = k.addk(gs, 32)?;
    let gs = k.shrk(gs, 6)?;
    let gt = k.mulk(t, dt)?;
    let gt = k.mul(gt, gh_less)?;
    let gt = k.addk(gt, 32)?;
    let gt = k.shrk(gt, 6)?;
    let js = k.shrk(gs, 4)?;
    let fs = k.andk(gs, 15)?;
    let jt = k.shrk(gt, 4)?;
    let ft = k.andk(gt, 15)?;
    let w11 = k.mul(fs, ft)?;
    let w11 = k.addk(w11, 8)?;
    let w11 = k.shrk(w11, 4)?;
    let w10 = k.sub(ft, w11)?;
    let w01 = k.sub(fs, w11)?;
    let sixteen = k.k(16)?;
    let w00 = k.sub(sixteen, fs)?;
    let w00 = k.sub(w00, ft)?;
    let w00 = k.add(w00, w11)?;
    let row = k.mul(jt, gw)?;
    let v0 = k.add(js, row)?;
    let v1 = k.addk(v0, 1)?;
    let v2 = k.add(v0, gw)?;
    let v3 = k.addk(v2, 1)?;
    let mut plane_weights = [zero; 2];
    for (plane, slot) in (0u32..).zip(plane_weights.iter_mut()) {
        let mut sum = k.k(8)?;
        for (index, factor) in [(v0, w00), (v1, w01), (v2, w10), (v3, w11)] {
            let at = k.mul(index, planes)?;
            let at = k.addk(at, plane)?;
            let raw = k.ise_value(&reversed, (zero, weight_bits), wise, at)?;
            let tables = (k.tables.weight_at, k.tables.weights);
            let weight = k.unquantize(tables, weight_range, raw)?;
            let inside = k.op(BinOp::ULessThan, at, weight_count)?;
            let weight = k.sel(inside, weight, zero)?;
            let term = k.mul(weight, factor)?;
            sum = k.add(sum, term)?;
        }
        *slot = k.shrk(sum, 4)?;
    }
    let second = k.sel(dual, plane_weights[1], plane_weights[0])?;
    let mut channels = [zero; 4];
    for (ch, slot) in (0u32..).zip(channels.iter_mut()) {
        let is_ccs = k.eqk(ccs, ch)?;
        let use_second = k.land(dual, is_ccs)?;
        let w = k.sel(use_second, second, plane_weights[0])?;
        let widen = |k: &mut K<'_, '_>, e: Value| match decode.srgb {
            true => {
                let up = k.shlk(e, 8)?;
                k.opk(BinOp::BitwiseOr, up, 0x80)
            }
            false => k.mulk(e, 257),
        };
        let mut c0 = widen(k, e0[ch as usize])?;
        let mut c1 = widen(k, e1[ch as usize])?;
        let mut lns = None;
        if let Some(([h0, h1], lns_alpha)) = hdr_ends {
            c0 = k.sel(hdr, h0[ch as usize], c0)?;
            c1 = k.sel(hdr, h1[ch as usize], c1)?;
            lns = Some(match ch {
                3 => k.land(hdr, lns_alpha)?,
                _ => hdr,
            });
        }
        let sixty_four = k.k(64)?;
        let rest = k.sub(sixty_four, w)?;
        let a = k.mul(c0, rest)?;
        let b = k.mul(c1, w)?;
        let sum = k.add(a, b)?;
        let sum = k.addk(sum, 32)?;
        let c = k.shrk(sum, 6)?;
        *slot = match lns {
            None => c,
            Some(lns) => {
                let log = k.lns_to_half(c)?;
                let linear = k.unorm16_to_half(c)?;
                k.sel(lns, log, linear)?
            }
        };
    }
    // A void-extent block (`super::void_extent`).
    let void_mode = k.andk(mode, 0x1ff)?;
    let void = k.eqk(void_mode, 0x1fc)?;
    let hdr_void = bit(k, 9)?;
    let hdr_void = k.opk(BinOp::INotEqual, hdr_void, 0)?;
    let reserved_void = k.field(w0, 10, 2)?;
    let reserved_void = k.opk(BinOp::INotEqual, reserved_void, 3)?;
    let mut coords = [zero; 4];
    let mut all_ones = k.boolean(true)?;
    for (i, slot) in (0u32..).zip(coords.iter_mut()) {
        let at = k.k(12 + 13 * i)?;
        let n = k.k(13)?;
        *slot = k.read(&words, at, n)?;
        let ones = k.eqk(*slot, 0x1fff)?;
        all_ones = k.land(all_ones, ones)?;
    }
    let empty_s = k.op(BinOp::UGreaterThanEqual, coords[0], coords[1])?;
    let empty_t = k.op(BinOp::UGreaterThanEqual, coords[2], coords[3])?;
    let empty = k.lor(empty_s, empty_t)?;
    let some = k.not(all_ones)?;
    let empty = k.land(some, empty)?;
    // An HDR format's void-extent block of HDR values reads them; an LDR or sRGB format's is an
    // error block.
    let void_bad = match decode.hdr {
        true => reserved_void,
        false => k.lor(hdr_void, reserved_void)?,
    };
    let void_bad = k.lor(void_bad, empty)?;
    let mut color = [zero; 4];
    for (i, slot) in color.iter_mut().enumerate() {
        let word = words[2 + i / 2];
        *slot = k.field(word, 16 * (i as u32 % 2), 16)?;
        if decode.hdr {
            let half = k.unorm16_to_half(*slot)?;
            *slot = k.sel(hdr_void, *slot, half)?;
        }
    }
    let channels = k.sel4(void, color, channels)?;
    let bad = k.sel(void, void_bad, partition_bad)?;
    Ok((channels, bad))
}

/// The module of `decode` writing `texels` as a typed module, before serialization: the module
/// [`astc_decode_module`](super::astc_decode_module) serializes, which the CPU interpreter runs.
/// Its bindings are the block decode's (`BlockDecodeModule`); it reads the shape's first three
/// words, the region's size.
pub(super) fn typed_module(decode: AstcDecode, texels: AstcTexels) -> BResult<TypedModule> {
    let mut b = ModuleBuilder::new();
    let u32_ = b.ty(Type::Int {
        width: 32,
        signed: false,
    })?;
    let uvec3 = b.ty(Type::Vector { elem: u32_, len: 3 })?;
    let shape = words_buffer(&mut b, SHAPE, u32_)?;
    let blocks = words_buffer(&mut b, BLOCKS, u32_)?;
    let texel_words = words_buffer(&mut b, TEXELS, u32_)?;
    let tables = Tables::new(&mut b, u32_)?;
    let invocation = b.builtin(crate::spv::BuiltIn::GlobalInvocationId, IoDir::Input, uvec3)?;
    let void = b.ty(Type::Void)?;
    let mut f = b.function(void)?;
    let zero = scalar(&mut f, u32_, 0)?;
    let invocation = f.module().global_value(invocation)?;
    let invocation = f.load(invocation, MemAccess::default())?;
    let x = f.composite_extract(invocation, &[0])?;
    let y = f.composite_extract(invocation, &[1])?;
    let z = f.composite_extract(invocation, &[2])?;
    let shape = f.module().global_value(shape)?;
    let field = |f: &mut FunctionBuilder<'_>, index: u32| -> BResult<Value> {
        let index = scalar(f, u32_, u64::from(index))?;
        let p = f.access_chain(shape, &[zero, index])?;
        f.load(p, MemAccess::default())
    };
    let width = field(&mut f, 0)?;
    let height = field(&mut f, 1)?;
    let depth = field(&mut f, 2)?;
    let in_x = f.binary(BinOp::ULessThan, x, width)?;
    let in_y = f.binary(BinOp::ULessThan, y, height)?;
    let in_z = f.binary(BinOp::ULessThan, z, depth)?;
    let inside = f.binary(BinOp::LogicalAnd, in_x, in_y)?;
    let inside = f.binary(BinOp::LogicalAnd, inside, in_z)?;
    let blocks = f.module().global_value(blocks)?;
    let texel_words = f.module().global_value(texel_words)?;
    let decoded: ArmFn<'_, ()> = Box::new(move |f, _| {
        let mut k = K { f, u32_, tables };
        let (bw, bh) = (decode.width, decode.height);
        // The texel's block, `(z · down + y / bh) · across + x / bw`, and its place in it.
        let across = k.addk(width, bw - 1)?;
        let across = k.opk(BinOp::UDiv, across, bw)?;
        let down = k.addk(height, bh - 1)?;
        let down = k.opk(BinOp::UDiv, down, bh)?;
        let bx = k.opk(BinOp::UDiv, x, bw)?;
        let by = k.opk(BinOp::UDiv, y, bh)?;
        let row = k.mul(z, down)?;
        let row = k.add(row, by)?;
        let block = k.mul(row, across)?;
        let block = k.add(block, bx)?;
        let base = k.shlk(block, 2)?;
        let mut words = [zero; 4];
        for (i, slot) in (0u32..).zip(words.iter_mut()) {
            let index = k.addk(base, i)?;
            let p = k.f.access_chain(blocks, &[zero, index])?;
            *slot = k.f.load(p, MemAccess::default())?;
        }
        let x0 = k.mulk(bx, bw)?;
        let s = k.sub(x, x0)?;
        let y0 = k.mulk(by, bh)?;
        let t = k.sub(y, y0)?;
        let (channels, bad) = texel(&mut k, decode, words, [s, t])?;
        // An error texel is magenta (`AstcDecode::error_texel`): each channel's `C` all ones, an
        // HDR format's FP16 1.0, green 0.
        let all = k.k(match decode.hdr {
            true => u32::from(HALF_ONE),
            false => 0xffff,
        })?;
        let channels = k.sel4(bad, [all, zero, all, all], channels)?;
        let out: Vec<Value> = match texels {
            AstcTexels::Unorm8 => {
                let mut packed = zero;
                for (ch, c) in (0u32..).zip(channels) {
                    let code = match (decode.hdr, decode.srgb) {
                        (true, _) => k.unorm8_of_half(c)?,
                        (false, true) => k.shrk(c, 8)?,
                        (false, false) => {
                            let twelve = k.shrk(c, 4)?;
                            let scaled = k.mulk(twelve, 255)?;
                            let scaled = k.addk(scaled, 2047)?;
                            k.opk(BinOp::UDiv, scaled, 4095)?
                        }
                    };
                    let placed = k.shlk(code, 8 * ch)?;
                    packed = k.or(packed, placed)?;
                }
                vec![packed]
            }
            AstcTexels::Half => {
                let mut halves = [zero; 4];
                for (ch, (slot, c)) in halves.iter_mut().zip(channels).enumerate() {
                    *slot = match (decode.hdr, decode.srgb, ch) {
                        (true, ..) => c,
                        (false, true, 3) => {
                            let code = k.shrk(c, 8)?;
                            k.half_of_ratio(code, 255)?
                        }
                        (false, true, _) => {
                            let code = k.shrk(c, 8)?;
                            let step = k.entry(k.tables.srgb_steps, code)?;
                            k.half_of_ratio(step, 4095)?
                        }
                        (false, false, _) => {
                            let twelve = k.shrk(c, 4)?;
                            k.half_of_ratio(twelve, 4095)?
                        }
                    };
                }
                let mut pairs = Vec::with_capacity(2);
                for pair in halves.chunks(2) {
                    let high = k.shlk(pair[1], 16)?;
                    pairs.push(k.or(pair[0], high)?);
                }
                pairs
            }
            AstcTexels::Float => {
                let u = Form {
                    held: Held::Int {
                        bits: 32,
                        container: 32,
                    },
                    lanes: 1,
                };
                let mut words = Vec::with_capacity(4);
                for (ch, c) in channels.into_iter().enumerate() {
                    if decode.hdr {
                        words.push(k.half_to_f32_bits(c)?);
                        continue;
                    }
                    let value = match (decode.srgb, ch) {
                        (true, 3) => {
                            let code = k.shrk(c, 8)?;
                            div_exact(k.f, u, code, 255)?
                        }
                        (true, _) => {
                            let code = k.shrk(c, 8)?;
                            let step = k.entry(k.tables.srgb_steps, code)?;
                            div_exact(k.f, u, step, 4095)?
                        }
                        (false, _) => {
                            let twelve = k.shrk(c, 4)?;
                            div_exact(k.f, u, twelve, 4095)?
                        }
                    };
                    words.push(k.f.convert(ConvOp::Bitcast, value, u32_)?);
                }
                words
            }
        };
        // The texel's first word, `((z · height + y) · width + x) · words`.
        let at = k.mul(z, height)?;
        let at = k.add(at, y)?;
        let at = k.mul(at, width)?;
        let at = k.add(at, x)?;
        let at = k.mulk(at, out.len() as u32)?;
        for (i, word) in (0u32..).zip(out) {
            let index = k.addk(at, i)?;
            let p = k.f.access_chain(texel_words, &[zero, index])?;
            k.f.store(p, word, MemAccess::default())?;
        }
        Ok(ArmEnd::FallThrough(Vec::new()))
    });
    f.if_(&mut (), inside, decoded, None)?;
    f.ret(None)?;
    let id = f.finish()?;
    b.entry(
        ExecModel::GLCompute,
        id,
        vec![ExecMode::LocalSize {
            x: BlockDecodeModule::LOCAL_SIZE,
            y: BlockDecodeModule::LOCAL_SIZE,
            z: 1,
        }],
    )?;
    b.finalize()
}
