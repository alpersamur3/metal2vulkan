use crate::memory::{
    candidates, host_cached, host_cached_placement, host_incoherent, invalidation_range,
    readback_candidates, readback_requirements, MemoryClass, Pool, KEPT_EMPTY_BLOCKS,
    KEPT_EMPTY_FOR,
};
use ash::vk;
use std::time::Instant;

fn ty(flags: vk::MemoryPropertyFlags) -> vk::MemoryType {
    vk::MemoryType {
        property_flags: flags,
        heap_index: 0,
    }
}

#[test]
fn classes_pick_their_memory_types_best_first() {
    use vk::MemoryPropertyFlags as P;
    let types = [
        ty(P::DEVICE_LOCAL),
        ty(P::HOST_VISIBLE | P::HOST_COHERENT),
        ty(P::DEVICE_LOCAL | P::HOST_VISIBLE | P::HOST_COHERENT),
        ty(P::HOST_VISIBLE | P::HOST_CACHED),
    ];
    let all = 0b1111;
    // Shared needs host-visible coherent memory and prefers host-cached, then device-local (the
    // cached type here is not coherent).
    assert_eq!(candidates(&types, all, MemoryClass::Shared, false), [2, 1]);
    // Private needs device-local memory and avoids host-visible.
    assert_eq!(candidates(&types, all, MemoryClass::Private, false), [0, 2]);
    // Only the types the resource allows.
    assert_eq!(candidates(&types, 0b0011, MemoryClass::Shared, false), [1]);
    assert!(candidates(&types, 0b1000, MemoryClass::Private, false).is_empty());
    // A device with only unified memory (lavapipe, Apple GPUs' shared types): both classes use it.
    let unified = [ty(P::DEVICE_LOCAL | P::HOST_VISIBLE | P::HOST_COHERENT)];
    assert_eq!(candidates(&unified, 1, MemoryClass::Shared, false), [0]);
    assert_eq!(candidates(&unified, 1, MemoryClass::Private, false), [0]);
    // Memoryless prefers lazily allocated memory, which no other class uses.
    let tiler = [
        ty(P::DEVICE_LOCAL),
        ty(P::DEVICE_LOCAL | P::LAZILY_ALLOCATED),
        ty(P::DEVICE_LOCAL | P::HOST_VISIBLE | P::HOST_COHERENT),
    ];
    assert_eq!(
        candidates(&tiler, 0b111, MemoryClass::Memoryless, false),
        [1, 0, 2]
    );
    assert_eq!(
        candidates(&tiler, 0b111, MemoryClass::Private, false),
        [0, 2]
    );
    // Without lazily allocated memory, as `Private`.
    assert_eq!(
        candidates(&types, all, MemoryClass::Memoryless, false),
        [0, 2]
    );
}

/// `Shared` memory prefers a host-cached coherent type, then a device-local one, with unified
/// memory or without; `WriteCombined` memory is `Shared` with unified memory and prefers a
/// device-local type without. A cached type that is not coherent is neither.
#[test]
fn shared_memory_follows_the_cpu_cache_mode_and_unified_memory() {
    use vk::MemoryPropertyFlags as P;
    let host = P::HOST_VISIBLE | P::HOST_COHERENT;
    let (shared, combined) = (MemoryClass::Shared, MemoryClass::WriteCombined);
    // RADV on a Radeon 780M: the cached type is not device-local.
    let radv = [
        ty(P::DEVICE_LOCAL),
        ty(host),
        ty(P::DEVICE_LOCAL | host),
        ty(host | P::HOST_CACHED),
    ];
    for unified in [true, false] {
        assert_eq!(candidates(&radv, 0b1111, shared, unified), [3, 2, 1]);
    }
    assert_eq!(candidates(&radv, 0b1111, combined, true), [3, 2, 1]);
    assert_eq!(candidates(&radv, 0b1111, combined, false), [2, 1, 3]);
    // ANV on an Iris Xe: every type is device-local.
    let anv = [
        ty(P::DEVICE_LOCAL),
        ty(P::DEVICE_LOCAL | host),
        ty(P::DEVICE_LOCAL | host | P::HOST_CACHED),
    ];
    for unified in [true, false] {
        assert_eq!(candidates(&anv, 0b111, shared, unified), [2, 1]);
    }
    assert_eq!(candidates(&anv, 0b111, combined, true), [2, 1]);
    assert_eq!(candidates(&anv, 0b111, combined, false), [1, 2]);
    // NVIDIA on an RTX 3090 (resizable BAR): system memory, cached or not, and BAR memory.
    let nvidia = [
        ty(P::empty()),
        ty(P::DEVICE_LOCAL),
        ty(P::DEVICE_LOCAL),
        ty(host),
        ty(host | P::HOST_CACHED),
        ty(P::DEVICE_LOCAL | host),
    ];
    assert_eq!(candidates(&nvidia, 0b11_1111, shared, false), [4, 5, 3]);
    assert_eq!(candidates(&nvidia, 0b11_1111, combined, false), [5, 3, 4]);
    // A cached type that is not coherent is never taken.
    let incoherent = [
        ty(P::DEVICE_LOCAL | host),
        ty(P::HOST_VISIBLE | P::HOST_CACHED),
    ];
    for class in [shared, combined] {
        for unified in [true, false] {
            assert_eq!(candidates(&incoherent, 0b11, class, unified), [0]);
        }
    }
    // The other classes do not depend on unified memory.
    for class in [
        MemoryClass::Private,
        MemoryClass::Managed,
        MemoryClass::Memoryless,
    ] {
        assert_eq!(
            candidates(&radv, 0b1111, class, true),
            candidates(&radv, 0b1111, class, false),
            "{class:?}"
        );
    }
}

/// No class takes a device-coherent or device-uncached type of `VK_AMD_device_coherent_memory`,
/// whose allocation needs a feature the runtime does not enable, even where it is the only type
/// the resource allows or ranks best by its other properties.
#[test]
fn no_class_takes_amd_device_coherent_memory() {
    use vk::MemoryPropertyFlags as P;
    let host = P::HOST_VISIBLE | P::HOST_COHERENT;
    let amd = P::DEVICE_COHERENT_AMD | P::DEVICE_UNCACHED_AMD;
    // RADV on a Radeon 780M lists them after its other types; here they come first as well.
    let types = [
        ty(P::DEVICE_LOCAL | host | P::HOST_CACHED | amd),
        ty(P::DEVICE_LOCAL),
        ty(host),
        ty(P::DEVICE_LOCAL | host),
        ty(host | P::HOST_CACHED),
        ty(P::DEVICE_LOCAL | amd),
        ty(P::DEVICE_LOCAL | host | P::DEVICE_COHERENT_AMD),
        ty(host | P::HOST_CACHED | P::DEVICE_UNCACHED_AMD),
    ];
    let all = (1 << types.len()) - 1;
    for unified in [false, true] {
        for class in [
            MemoryClass::Shared,
            MemoryClass::WriteCombined,
            MemoryClass::Private,
            MemoryClass::Managed,
            MemoryClass::Memoryless,
        ] {
            let picked = candidates(&types, all, class, unified);
            assert!(!picked.is_empty(), "{class:?}");
            for index in picked {
                assert!(
                    !types[index as usize].property_flags.intersects(amd),
                    "{class:?} took type {index}"
                );
            }
            assert!(
                candidates(&types, 0b1110_0001, class, unified).is_empty(),
                "{class:?}"
            );
        }
    }
}

/// A texture's image with host-transfer usage takes host-cached coherent memory of its class
/// (`Private`) where the device has it among the image's types, else, with unified memory,
/// host-cached `Shared` memory where the device has it among them, else the class's memory as any
/// image does (a copy through a submission could queue behind the program's work), never a type
/// the host maps without coherence (ANV's cached type on Arrow Lake-S). On the memory types and
/// host-transfer image memory type bits measured, the RTX 3090 has none, so its images take
/// resizable BAR, and the Radeon 780M (unified) takes its host-cached
/// system memory, type 5, where it took RADV's uncached device-local type before. MoltenVK's shared
/// type on Apple GPUs (`MVK_VK_MEMORY_TYPE_METAL_SHARED` with `DEVICE_LOCAL`, MoltenVK 1.4.1) and
/// lavapipe's one type are host-cached and device-local; a device with both kinds of device-local
/// host-visible memory takes the cached one where `Private`'s ranking alone would take the first.
#[test]
fn host_transfer_images_prefer_host_cached_memory() {
    let types = |raw: &[u32]| -> Vec<vk::MemoryType> {
        raw.iter()
            .map(|&flags| ty(vk::MemoryPropertyFlags::from_raw(flags)))
            .collect()
    };
    let private = MemoryClass::Private;
    // The class the image takes and the types it may lie in, best first.
    let picked = |types: &[vk::MemoryType], allowed: u32, unified: bool| {
        let (class, taken) =
            host_cached_placement(types, allowed, private, unified).unwrap_or((private, allowed));
        (class, candidates(types, taken, class, unified))
    };
    // NVIDIA 615.71.09 on an RTX 3090: a host-transfer image may lie in types 3 to 5.
    let nvidia = types(&[0x0, 0x1, 0x1, 0x6, 0xe, 0x7]);
    assert_eq!(host_cached(&nvidia), 0b1_0000);
    assert_eq!(
        host_cached_placement(&nvidia, 0b11_1000, private, false),
        None
    );
    assert_eq!(picked(&nvidia, 0b11_1000, false), (private, vec![5]));
    // RADV (Mesa 26.2.4) on a Radeon 780M: types 2, 3, 5 and the AMD device-coherent 14 to 16.
    let radv = types(&[
        0x1, 0x1, 0x6, 0x7, 0x7, 0xe, 0xe, 0x21, 0x21, 0x21, 0x21, 0x20, 0x20, 0xc1, 0xc6, 0xc7,
        0xce,
    ]);
    let allowed = 0b1_1100_0000_0010_1100;
    assert_eq!(picked(&radv, allowed, true), (MemoryClass::Shared, vec![5]));
    // The same memory on a device without unified memory: its device-local uncached type.
    assert_eq!(host_cached_placement(&radv, allowed, private, false), None);
    assert_eq!(picked(&radv, allowed, false), (private, vec![3]));
    // An image that may lie in no host-cached type keeps its class's memory.
    assert_eq!(picked(&radv, 0b1000, true), (private, vec![3]));
    // MoltenVK on an Apple GPU: private, shared (device-local and host-cached), memoryless.
    let moltenvk = types(&[0x1, 0xf, 0x11]);
    assert_eq!(host_cached(&moltenvk), 0b010);
    assert_eq!(picked(&moltenvk, 0b111, true), (private, vec![1]));
    // lavapipe: one type, device-local and host-cached.
    assert_eq!(picked(&types(&[0xf]), 1, true), (private, vec![0]));
    // Device-local host-visible memory both uncached (listed first) and cached.
    let both = types(&[0x1, 0x7, 0xf]);
    assert_eq!(candidates(&both, 0b110, private, true), [1, 2]);
    assert_eq!(picked(&both, 0b110, true), (private, vec![2]));
    // ANV (Mesa 26.2.4) on Arrow Lake-S's Xe-LPG (0x7d67), which shares no cache with the host:
    // coherent write-combined types 0 and 3, cached type 1 without coherence, protected type 2.
    // No type is host-cached and coherent, so the image takes its class's memory, without the
    // incoherent type (`placed_image` strips it): the coherent type 0 first.
    let arl = types(&[0x7, 0xb, 0x21, 0x7]);
    assert_eq!(host_cached(&arl), 0);
    assert_eq!(host_incoherent(&arl), 0b0010);
    assert_eq!(host_cached_placement(&arl, 0b1011, private, true), None);
    let coherent = 0b1011 & !host_incoherent(&arl);
    assert_eq!(picked(&arl, coherent, true), (private, vec![0, 3]));
    // Every device above lists only coherent host-visible types.
    for listed in [nvidia, radv, moltenvk, types(&[0xf]), both] {
        assert_eq!(host_incoherent(&listed), 0);
    }
}

/// The memory log's line for a large pooled placement names what it is, its class where placed,
/// its bytes and where it lies, in the words of the block's own lines (`memory_log.rs`).
#[test]
fn the_memory_log_names_a_placement_s_kind_class_and_place() {
    use crate::memory_log::placement_line;
    use ash::vk::Handle;
    let memory = vk::DeviceMemory::from_raw(0xabc0);
    let placed = placement_line(
        "placed",
        16 << 20,
        Some(MemoryClass::Private),
        true,
        (memory, 0x200_0000, 0),
    );
    assert_eq!(
        placed,
        "metal2vulkan: memory: placed a Private image of 16777216 bytes at offset 0x2000000 of \
         memory 0xabc0 of type 0"
    );
    let returned = placement_line("returned", 4 << 20, None, false, (memory, 0, 2));
    assert_eq!(
        returned,
        "metal2vulkan: memory: returned a buffer of 4194304 bytes at offset 0x0 of memory 0xabc0 \
         of type 2"
    );
}

/// Pooled blocks take placements until full; a block that empties is kept while fewer than
/// `KEPT_EMPTY_BLOCKS` empty ones are, and freed past that, so a working set that dips by a block
/// and comes back makes and frees none. Freed slots are reused.
#[test]
fn pooled_blocks_place_until_full_and_two_empty_blocks_are_kept() {
    assert_eq!(KEPT_EMPTY_BLOCKS, 2);
    let now = Instant::now();
    let mut pool: Pool<&str> = Pool::default();
    assert_eq!(pool.place(16, 16), None, "no block yet");
    let first = pool.add(256, false, "first", 100);
    assert_eq!(pool.place(100, 64), Some((first, 128)));
    assert_eq!(pool.place(100, 64), None, "full");
    let second = pool.add(256, false, "second", 200);
    let third = pool.add(256, false, "third", 256);
    assert_ne!(first, second);
    assert_eq!(pool.blocks(), 3);
    // Emptying the first block keeps it, and the second likewise: two empty blocks are kept.
    assert_eq!(pool.release(first, 0, 100, now), None);
    assert_eq!(pool.release(first, 128, 100, now), None);
    assert_eq!(pool.release(second, 0, 200, now), None);
    assert_eq!(pool.blocks(), 3);
    // Emptying the third while two are kept frees it.
    assert_eq!(pool.release(third, 0, 256, now), Some("third"));
    assert_eq!(pool.blocks(), 2);
    // The kept blocks serve the next requests, the first-made first; the freed slot is reused.
    assert_eq!(pool.place(256, 1), Some((first, 0)));
    assert_eq!(pool.place(256, 1), Some((second, 0)));
    let fourth = pool.add(256, false, "fourth", 1);
    assert_eq!(fourth, third, "the freed slot is reused");
    assert_eq!(pool.turnover(), (4, 1));
    let mut drained: Vec<&str> = pool.drain().collect();
    drained.sort_unstable();
    assert_eq!(drained, ["first", "fourth", "second"]);
    assert_eq!(pool.blocks(), 0);
}

/// A kept empty block beyond its pool's one goes once it stayed empty for `KEPT_EMPTY_FOR`, the
/// one emptied earliest first, and never the last kept one; a placement in a kept block makes it
/// a block in use again. Every kept block goes where the caller takes them (a block of the type
/// failing to be made).
#[test]
fn a_kept_empty_block_beyond_one_goes_after_its_quiet_period() {
    let start = Instant::now();
    let later = start + KEPT_EMPTY_FOR;
    let mut pool: Pool<&str> = Pool::default();
    let first = pool.add(256, false, "first", 256);
    let second = pool.add(256, false, "second", 256);
    assert_eq!(pool.release(first, 0, 256, start), None);
    assert_eq!(
        pool.stale(later),
        None,
        "one kept block stays, however long"
    );
    assert_eq!(
        pool.release(second, 0, 256, start + KEPT_EMPTY_FOR / 2),
        None
    );
    assert_eq!(
        pool.stale(later - KEPT_EMPTY_FOR / 4),
        None,
        "neither stayed empty so long"
    );
    assert_eq!(
        pool.stale(later),
        Some("first"),
        "the earliest emptied goes first"
    );
    assert_eq!(
        pool.stale(later + KEPT_EMPTY_FOR),
        None,
        "the last kept block stays"
    );
    assert_eq!(pool.blocks(), 1);
    // Placed in again, a kept block is no longer empty, and emptied again it starts anew.
    let third = pool.add(256, false, "third", 256);
    assert_eq!(pool.place(256, 1), Some((second, 0)));
    let at = later + KEPT_EMPTY_FOR;
    assert_eq!(pool.release(second, 0, 256, at), None);
    assert_eq!(pool.release(third, 0, 256, at), None);
    assert_eq!(pool.stale(at + KEPT_EMPTY_FOR / 2), None);
    let mut taken = pool.take_kept();
    taken.sort_unstable();
    assert_eq!(taken, ["second", "third"]);
    assert_eq!(pool.blocks(), 0);
    assert_eq!(pool.turnover(), (3, 3));
}

#[test]
fn dedicated_blocks_take_no_other_placement_and_free_with_their_resource() {
    let mut pool: Pool<u32> = Pool::default();
    let dedicated = pool.add(1000, true, 7, 1000);
    assert_eq!(pool.place(1, 1), None, "a dedicated block is never shared");
    assert_eq!(pool.release(dedicated, 0, 1000, Instant::now()), Some(7));
    assert_eq!(pool.blocks(), 0);
}

/// A pool has room for a placement where a pooled block holds a free range that long at the
/// largest alignment asked of the pool so far; a dedicated block's room counts for none.
#[test]
fn room_is_a_pooled_block_s_free_range_at_the_largest_alignment_asked() {
    let mut pool: Pool<u32> = Pool::default();
    assert!(!pool.has_room(1), "no block yet");
    pool.add(1000, true, 1, 10);
    assert!(
        !pool.has_room(1),
        "a dedicated block's free bytes are no room"
    );
    let block = pool.add(256, false, 2, 0);
    assert!(pool.has_room(256), "an empty block");
    assert_eq!(pool.place(8, 1), Some((block, 0)));
    assert!(pool.has_room(248) && !pool.has_room(249));
    // A placement aligned to 64 asks the pool's later room at 64: 192 bytes from offset 64.
    assert_eq!(pool.place(8, 64), Some((block, 64)));
    assert!(pool.has_room(128) && !pool.has_room(129));
    assert_eq!(pool.release(block, 64, 8, Instant::now()), None);
    assert!(pool.has_room(192) && !pool.has_room(193));
}

#[test]
fn buffers_are_placed_at_the_largest_linear_image_alignment() {
    use crate::buffer::{placed, LINEAR_IMAGE_ALIGNMENT};
    let requirements = |alignment| vk::MemoryRequirements {
        size: 100,
        alignment,
        memory_type_bits: 0b101,
    };
    // Each linear-image alignment measured divides it (RADV, NVIDIA, lavapipe, MoltenVK).
    for measured in [256, 128, 64, 16] {
        assert!(LINEAR_IMAGE_ALIGNMENT.is_multiple_of(measured));
    }
    // A smaller alignment is raised to it, a larger one is kept, and the size takes whole units
    // of the alignment; the memory types are kept.
    for (alignment, placed_at) in [(1, 256), (16, 256), (256, 256), (4096, 4096)] {
        let placed = placed(requirements(alignment));
        assert_eq!(placed.alignment, placed_at);
        assert_eq!((placed.size, placed.memory_type_bits), (placed_at, 0b101));
    }
}

/// Buffers shorter than their placement's alignment, placed one after another, leave no gap:
/// each starts where the last ended, so the block's free ranges stay one and the pool's
/// first-fit search does not walk a gap per buffer placed before (65,537 buffers of 16 bytes
/// took 201 s to make on lavapipe, each later one longer).
#[test]
fn short_buffers_are_placed_without_gaps() {
    use crate::buffer::placed;
    let mut pool: Pool<()> = Pool::default();
    let block = pool.add(1 << 20, false, (), 0);
    let mut end = 0;
    for length in [16, 1, 100, 255, 256, 257, 16, 4000] {
        let requirements = placed(vk::MemoryRequirements {
            size: length,
            alignment: 16,
            memory_type_bits: 1,
        });
        let (at, offset) = pool
            .place(requirements.size, requirements.alignment)
            .unwrap();
        assert_eq!((at, offset), (block, end), "{length} bytes");
        end = offset + requirements.size;
    }
}

/// Memory types as a GPU check's report lists them (`memory type N: flags 0x..`).
fn listed(raw: &[u32]) -> Vec<vk::MemoryType> {
    raw.iter()
        .map(|&flags| ty(vk::MemoryPropertyFlags::from_raw(flags)))
        .collect()
}

/// Where the device has a memory type the host caches and keeps coherent, a read-back buffer takes
/// exactly the types `Shared` memory does, in the same order, whatever types its buffer allows
/// and with or without unified memory: nothing changes on those devices.
#[test]
fn readback_memory_is_shared_memory_where_a_cached_coherent_type_exists() {
    let want = vk::MemoryPropertyFlags::HOST_VISIBLE
        | vk::MemoryPropertyFlags::HOST_COHERENT
        | vk::MemoryPropertyFlags::HOST_CACHED;
    let devices: [(&str, Vec<vk::MemoryType>); 10] = [
        // Turnip on an Adreno 830 (Galaxy S25 Ultra, GPU check report of issue 15).
        ("Adreno 830", listed(&[0x7, 0xf, 0xb, 0x11])),
        // Turnip on an Adreno 825 (Poco F7, issue 4); 750 and 740 list the same shape.
        (
            "Adreno 825",
            listed(&[0x1, 0x1, 0x1, 0xb, 0x7, 0x7, 0xf, 0x21, 0x11]),
        ),
        // Samsung's driver on an Xclipse 550 (issue 62): two copies of the same four types.
        (
            "Xclipse 550",
            listed(&[0x7, 0xf, 0x1, 0x21, 0x7, 0xf, 0x1, 0x21]),
        ),
        // RADV on a Radeon 780M, with the AMD device-coherent types 14 to 16.
        (
            "RADV 780M",
            listed(&[
                0x1, 0x1, 0x6, 0x7, 0x7, 0xe, 0xe, 0x21, 0x21, 0x21, 0x21, 0x20, 0x20, 0xc1, 0xc6,
                0xc7, 0xce,
            ]),
        ),
        // ANV on an Iris Xe: every type device-local.
        ("ANV Iris Xe", listed(&[0x1, 0x7, 0xf])),
        // NVIDIA on an RTX 3090.
        ("RTX 3090", listed(&[0x0, 0x1, 0x1, 0x6, 0xe, 0x7])),
        ("MoltenVK", listed(&[0x1, 0xf, 0x11])),
        ("lavapipe", listed(&[0xf])),
        // A cached coherent type beside an incoherent one.
        ("cached both ways", listed(&[0x7, 0xb, 0xf])),
        ("cached coherent last", listed(&[0x1, 0x6, 0xb, 0xe])),
    ];
    for (name, types) in &devices {
        for allowed in 0..1u32 << types.len() {
            for unified in [false, true] {
                let shared = candidates(types, allowed, MemoryClass::Shared, unified);
                let got = candidates(types, allowed, MemoryClass::Readback, unified);
                let cached_coherent = shared
                    .iter()
                    .any(|&index| types[index as usize].property_flags.contains(want));
                if cached_coherent {
                    assert_eq!(got, shared, "{name} allowed {allowed:#b} unified {unified}");
                } else {
                    // Only incoherent cached types may come first; the rest is Shared's.
                    assert!(got.ends_with(&shared), "{name} allowed {allowed:#b}");
                }
            }
        }
    }
    // On the devices that list a cached coherent type among all their types, the first pick is the
    // one `Shared` makes today.
    for (name, first) in [
        ("Adreno 830", 1),
        ("Xclipse 550", 1),
        ("ANV Iris Xe", 2),
        ("RTX 3090", 4),
    ] {
        let types = &devices.iter().find(|(n, _)| *n == name).unwrap().1;
        let all = (1 << types.len()) - 1;
        let got = candidates(types, all, MemoryClass::Readback, true);
        assert_eq!(got[0], first, "{name}");
        assert_eq!(got, candidates(types, all, MemoryClass::Shared, true));
    }
}

/// Where the only host-cached type is not coherent (ARM Mali, PowerVR: types 0x7 uncached and 0xb
/// cached, from the GPU check reports of issues 51, 52 and 64), a read-back buffer takes the
/// cached type first and `Shared`'s after it, so that its allocation can still fall back.
#[test]
fn readback_memory_takes_a_cached_incoherent_type_first_where_there_is_no_cached_coherent_one() {
    // Mali-G720 MC7 (MT6899, issue 51): 0x11 is lazily allocated, 0x21 protected.
    let g720 = listed(&[0x7, 0xb, 0x11, 0x21]);
    assert_eq!(candidates(&g720, 0b1111, MemoryClass::Shared, true), [0]);
    assert_eq!(
        candidates(&g720, 0b1111, MemoryClass::Readback, true),
        [1, 0]
    );
    // Mali-G57 (UMS9230H, issue 52).
    let g57 = listed(&[0x7, 0xb, 0x11]);
    for unified in [false, true] {
        assert_eq!(candidates(&g57, 0b111, MemoryClass::Shared, unified), [0]);
        assert_eq!(
            candidates(&g57, 0b111, MemoryClass::Readback, unified),
            [1, 0]
        );
    }
    // PowerVR BXM-8-256 (MT6855, issue 64): the cached type is the last.
    let powervr = listed(&[0x1, 0x11, 0x7, 0xb]);
    assert_eq!(candidates(&powervr, 0b1111, MemoryClass::Shared, true), [2]);
    assert_eq!(
        candidates(&powervr, 0b1111, MemoryClass::Readback, true),
        [3, 2]
    );
    // A buffer that may not lie in the cached type takes what Shared does.
    assert_eq!(candidates(&g720, 0b0001, MemoryClass::Readback, true), [0]);
    assert_eq!(candidates(&g720, 0b1101, MemoryClass::Readback, true), [0]);
    // ANV on Arrow Lake-S's Xe-LPG lists the same shape (0x7, 0xb, protected 0x21, 0x7): its
    // buffer is the one case outside the Mali and PowerVR lists that now takes the cached type.
    let arl = listed(&[0x7, 0xb, 0x21, 0x7]);
    assert_eq!(
        candidates(&arl, 0b1111, MemoryClass::Readback, true),
        [1, 0, 3]
    );
    // Never taken: a lazily allocated or AMD device-coherent type with the cached bits, nor a
    // type the host cannot see.
    let cached = vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_CACHED;
    let odd = [
        ty(vk::MemoryPropertyFlags::DEVICE_LOCAL
            | vk::MemoryPropertyFlags::HOST_VISIBLE
            | vk::MemoryPropertyFlags::HOST_COHERENT),
        ty(cached | vk::MemoryPropertyFlags::LAZILY_ALLOCATED),
        ty(cached | vk::MemoryPropertyFlags::DEVICE_COHERENT_AMD),
        ty(vk::MemoryPropertyFlags::DEVICE_LOCAL | vk::MemoryPropertyFlags::HOST_CACHED),
    ];
    assert_eq!(candidates(&odd, 0b1111, MemoryClass::Readback, true), [0]);
    // No coherent type at all: the cached one alone, where Shared has none.
    let alone = listed(&[0xa]);
    assert_eq!(readback_candidates(&alone, 1, true), [0]);
    assert!(candidates(&alone, 1, MemoryClass::Shared, true).is_empty());
    // No host-visible type: none, as for Shared.
    assert!(candidates(&listed(&[0x1]), 1, MemoryClass::Readback, true).is_empty());
}

/// Images and every other class keep their rule: the cached incoherent type is no image's
/// (`host_cached`, `host_incoherent`), and no other class takes it.
#[test]
fn only_a_readback_buffer_takes_a_cached_incoherent_type() {
    let g720 = listed(&[0x7, 0xb, 0x11, 0x21]);
    assert_eq!(host_cached(&g720), 0);
    assert_eq!(host_incoherent(&g720), 0b0010);
    // The classes that require coherent memory never take it. `Private` and `Memoryless` need
    // device-local memory and only rank host-visible types last, as before: their code is not
    // touched.
    for class in [
        MemoryClass::Shared,
        MemoryClass::WriteCombined,
        MemoryClass::Managed,
    ] {
        for unified in [false, true] {
            assert!(
                !candidates(&g720, 0b1111, class, unified).contains(&1),
                "{class:?}"
            );
        }
    }
}

/// An invalidated range is whole atoms, covering the bytes read; a read-back buffer's placement,
/// asked at a multiple of the atom, needs no rounding.
#[test]
fn an_invalidated_range_is_whole_atoms_covering_the_bytes() {
    // The device's `nonCoherentAtomSize` is 64 on Mali, Adreno and RADV, 256 on some others.
    assert_eq!(invalidation_range(0, 100, 64), (0, 128));
    assert_eq!(invalidation_range(192, 64, 64), (192, 64));
    assert_eq!(invalidation_range(200, 100, 64), (192, 128));
    assert_eq!(invalidation_range(7, 5, 1), (7, 5));
    assert_eq!(invalidation_range(7, 5, 0), (7, 5));
    assert_eq!(invalidation_range(1 << 20, 1, 256), (1 << 20, 256));
    for atom in [1u64, 4, 64, 128, 256] {
        for offset in [0u64, 1, 63, 64, 65, 4096, 4097] {
            for len in [1u64, 63, 64, 65, 1000, 24_110_592] {
                let (start, size) = invalidation_range(offset, len, atom);
                assert_eq!(start % atom, 0);
                assert_eq!(size % atom, 0);
                assert!(start <= offset && start + size >= offset + len);
                assert!(start + size < offset + len + atom && offset - start < atom);
            }
        }
    }
    // A placement at the aligned size and alignment lies in whole atoms.
    let wanted = vk::MemoryRequirements {
        size: 24_110_100,
        alignment: 16,
        memory_type_bits: 0b11,
    };
    let aligned = readback_requirements(&wanted, 64);
    assert_eq!(aligned.size % 64, 0);
    assert!(aligned.size >= wanted.size && aligned.size < wanted.size + 64);
    assert_eq!(aligned.alignment, 64);
    assert_eq!(aligned.memory_type_bits, 0b11);
    let wide = vk::MemoryRequirements {
        alignment: 128,
        ..wanted
    };
    assert_eq!(readback_requirements(&wide, 64).alignment, 128);
    assert_eq!(readback_requirements(&wide, 256).alignment, 256);
    assert_eq!(readback_requirements(&wanted, 0).size, wanted.size);
    // At any offset the placement aligns to, the range is the placement's own.
    let at = invalidation_range(aligned.alignment * 3, wanted.size, 64);
    assert_eq!(at, (192, aligned.size));
}
