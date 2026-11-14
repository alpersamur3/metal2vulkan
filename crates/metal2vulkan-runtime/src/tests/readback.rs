//! The device's kept read-back buffer (`readback.rs`): it is kept up to a bound and shares the
//! spare-block rule of `generate_mipmaps`' scratch (`mipmap_scratch.rs`).

use super::super::mipmap_scratch::{Spare, Take};
use super::super::readback::{
    announce, host_copy_line, keeps, read_back_buffer_line, ReadPath, KEPT_MAX,
};
use ash::vk::MemoryPropertyFlags as F;

#[test]
fn the_line_names_a_coherent_cached_type() {
    let flags = F::HOST_VISIBLE | F::HOST_COHERENT | F::HOST_CACHED;
    assert_eq!(
        read_back_buffer_line(12052800, 1, flags, true),
        "m2v: read-back buffer: 12052800 bytes on memory type 1 \
         (HOST_VISIBLE | HOST_COHERENT | HOST_CACHED), coherent"
    );
}

#[test]
fn the_line_names_a_cached_type_that_is_invalidated() {
    let flags = F::HOST_VISIBLE | F::HOST_CACHED;
    assert_eq!(
        read_back_buffer_line(12052800, 3, flags, false),
        "m2v: read-back buffer: 12052800 bytes on memory type 3 \
         (HOST_VISIBLE | HOST_CACHED), invalidated after each copy"
    );
}

#[test]
fn the_line_says_so_for_a_type_with_no_known_flag() {
    let line = read_back_buffer_line(4096, 0, F::empty(), false);
    assert_eq!(
        line,
        "m2v: read-back buffer: 4096 bytes on memory type 0 (no known flag), \
         invalidated after each copy"
    );
    assert!(line.is_ascii() && !line.contains('\n'));
}

#[test]
fn the_line_lists_every_flag_in_a_fixed_order() {
    let flags = F::DEVICE_LOCAL | F::HOST_VISIBLE | F::LAZILY_ALLOCATED | F::PROTECTED;
    assert_eq!(
        read_back_buffer_line(1, 7, flags, true),
        "m2v: read-back buffer: 1 bytes on memory type 7 \
         (HOST_VISIBLE | DEVICE_LOCAL | LAZILY_ALLOCATED | PROTECTED), coherent"
    );
}

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

#[test]
fn the_host_copy_line_names_a_cached_type() {
    let flags = F::HOST_VISIBLE | F::HOST_COHERENT | F::HOST_CACHED | F::DEVICE_LOCAL;
    assert_eq!(
        host_copy_line(1, flags),
        "m2v: read-back: host image copy from memory type 1 \
         (HOST_VISIBLE | HOST_COHERENT | HOST_CACHED | DEVICE_LOCAL)"
    );
}

#[test]
fn the_host_copy_line_marks_an_uncached_type() {
    let flags = F::HOST_VISIBLE | F::HOST_COHERENT | F::DEVICE_LOCAL;
    assert_eq!(
        host_copy_line(0, flags),
        "m2v: read-back: host image copy from memory type 0 \
         (HOST_VISIBLE | HOST_COHERENT | DEVICE_LOCAL), uncached"
    );
}

#[test]
fn each_path_and_type_is_announced_once() {
    // Types no other test uses: the record is process-wide.
    assert!(announce(ReadPath::HostCopy, 200, true));
    assert!(!announce(ReadPath::HostCopy, 200, true));
    assert!(announce(ReadPath::Buffer, 200, true));
    assert!(announce(ReadPath::HostCopy, 200, false));
    assert!(announce(ReadPath::HostCopy, 201, true));
    assert!(!announce(ReadPath::Buffer, 200, true));
}
