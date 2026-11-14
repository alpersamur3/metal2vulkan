//! The device's kept read-back buffer (`readback.rs`): it is kept up to a bound and shares the
//! spare-block rule of `generate_mipmaps`' scratch (`mipmap_scratch.rs`).

use super::super::mipmap_scratch::{Spare, Take};
use super::super::readback::{keeps, KEPT_MAX};

/// Block `id` of `size` bytes, as a test's stand-in.
type Block = (u32, u64);

/// A frame is read back again and again through one buffer; a larger read replaces it, and a read
/// that finds it leased by another thread makes its own, of which the larger is kept.
#[test]
fn a_frame_loop_reuses_one_buffer_and_a_larger_read_replaces_it() {
    let mut kept: Spare<Block> = Spare::default();
    let frame = 24 << 20;
    // The first read makes it, the next 1000 reuse it.
    let Take::Make(None) = kept.take(frame) else {
        panic!("the first read makes a buffer");
    };
    assert_eq!(kept.give((1, frame), frame), None);
    for _ in 0..1000 {
        let Take::Reuse(block, size) = kept.take(frame) else {
            panic!("a frame reuses the buffer");
        };
        assert_eq!((block, size), ((1, frame), frame));
        assert_eq!(kept.give(block, size), None);
    }
    // A small read (the guest's own getBytes) reuses it too.
    let Take::Reuse(block, size) = kept.take(4096) else {
        panic!("a small read reuses the buffer");
    };
    // Meanwhile another thread's read finds none, makes its own, and gives it back: the one
    // given back second is the smaller (here equal) and is released.
    let Take::Make(None) = kept.take(frame) else {
        panic!("a read of a leased buffer makes its own");
    };
    assert_eq!(kept.give((2, frame), frame), None);
    assert_eq!(kept.give(block, size), Some((1, frame)));
    // A larger frame (another resolution after a load) releases the kept one and keeps the new.
    let bigger = 40 << 20;
    let Take::Make(Some(stale)) = kept.take(bigger) else {
        panic!("a larger read releases the buffer too small");
    };
    assert_eq!(stale, (2, frame));
    assert_eq!(kept.give((3, bigger), bigger), None);
    assert!(matches!(kept.take(frame), Take::Reuse((3, _), _)));
}

/// The device releases the idle buffer as it is destroyed: `drain` gives it once.
#[test]
fn the_idle_buffer_is_drained_once() {
    let mut kept: Spare<Block> = Spare::default();
    assert_eq!(kept.drain(), None);
    assert_eq!(kept.give((1, 4096), 4096), None);
    assert_eq!(kept.drain(), Some((1, 4096)));
    assert_eq!(kept.drain(), None);
}

#[test]
fn a_buffer_larger_than_the_bound_is_not_kept() {
    assert!(keeps(1));
    assert!(keeps(24 << 20));
    // A 4K display's frame at 8 bytes a pixel is kept.
    assert!(keeps(3840 * 2160 * 8));
    assert!(keeps(KEPT_MAX));
    assert!(!keeps(KEPT_MAX + 1));
    assert!(!keeps(512 << 20));
}
