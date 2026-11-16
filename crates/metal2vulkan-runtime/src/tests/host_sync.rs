//! Cache maintenance for host imports in memory the GPU is not coherent with (`host_sync.rs`).
//!
//! The pure parts (the mode, the span of a range) run everywhere. The device tests run on the
//! device the cfg selects. Every host-visible type of lavapipe, where they run here, is coherent,
//! so each test makes its imports count as ones in a type that is not (`host_sync::forced`): the
//! import is then mapped whole, planned, flushed and invalidated as one in a Mali's cached type
//! would be (that the calls are made, with the right ranges, at the right moments, is what lavapipe
//! can show; that a real Mali's caches behave as Vulkan says is not). The tests check:
//! - a buffer the GPU writes is flushed once at its command buffer's first submission and, cleaned
//!   once more, invalidated once at its completion; one it only reads is flushed and never
//!   invalidated; a command buffer that uses neither does neither; the bytes of each range;
//! - the mode (`flush`, `off`) and the device that never made such an import (a coherent one)
//!   count nothing and keep the maintenance switched off;
//! - a map that fails leaves the import usable and unsynced, with a note;
//! - a heap over host memory, a rebind and two imports in one command buffer.
//!
//! A device test: compiled only under the validation crate's GPU safety cfgs (`m2v_gpu`,
//! `m2v_cpu_icd`), on Linux, on the kind of device the cfg selects.

use crate::host_sync::{Mode, Span};
use ash::vk;

#[test]
fn the_mode_is_full_unless_the_variable_says_flush_or_off() {
    assert_eq!(Mode::parse(None), Mode::Full);
    assert_eq!(Mode::parse(Some("")), Mode::Full);
    assert_eq!(Mode::parse(Some("full")), Mode::Full);
    assert_eq!(Mode::parse(Some("FULL")), Mode::Full);
    assert_eq!(Mode::parse(Some(" flush ")), Mode::Flush);
    assert_eq!(Mode::parse(Some("Flush")), Mode::Flush);
    for off in ["off", "OFF", "0", "no", "none", "false", " off\n"] {
        assert_eq!(Mode::parse(Some(off)), Mode::Off, "{off:?}");
    }
    // What is not understood keeps the maintenance on: the safe choice.
    assert_eq!(Mode::parse(Some("flsuh")), Mode::Full);
    assert_eq!(Mode::parse(Some("1")), Mode::Full);
    assert_eq!(Mode::parse(Some("on")), Mode::Full);
    // What each mode does.
    assert!(Mode::Full.flushes() && Mode::Full.invalidates());
    assert!(Mode::Flush.flushes() && !Mode::Flush.invalidates());
    assert!(!Mode::Off.flushes() && !Mode::Off.invalidates());
    for mode in [Mode::Full, Mode::Flush, Mode::Off] {
        assert_eq!(Mode::parse(Some(mode.name())), mode);
    }
}

#[test]
fn a_span_is_whole_atoms_of_a_buffer_s_reach_and_none_where_it_is_empty() {
    let memory = vk::DeviceMemory::null();
    // A buffer 0x123 bytes into a page, 0x1800 long, its reach the rest of the 8 KiB window, on a
    // device with an atom of 64: from the atom its first byte lies in to the memory's end.
    let span = Span::of(memory, (0x123 - 35, 0x2000 - 0x123 + 35), (64, 0x2000)).unwrap();
    assert_eq!((span.offset, span.size), (0x100, 0x1f00));
    // The last atom of the memory, cut where the memory ends.
    let span = Span::of(memory, (0x1ff0, 8), (64, 0x2000 - 8)).unwrap();
    assert_eq!((span.offset, span.size), (0x1fc0, 0x38));
    assert!(Span::of(memory, (10, 0), (64, 4096)).is_none());
    assert!(Span::of(memory, (5000, 8), (64, 4096)).is_none());
}

#[cfg(all(
    any(m2v_gpu, m2v_cpu_icd),
    any(target_os = "linux", target_os = "android")
))]
mod device {
    use super::super::host_residency::{counting, device, Scratch};
    use crate::host_import_note::take_host_import_notes;
    use crate::host_sync::{forced, Mode, Span};
    use crate::{
        Buffer, CommandQueue, Device, HeapDescriptor, HeapType, HostMemory, ResourceOptions,
    };
    use std::ptr::NonNull;
    use std::time::Duration;

    const TEST: &str = "host sync";

    /// The longest a command buffer may take.
    const WAIT: Duration = Duration::from_secs(60);

    /// Bytes of a test's pages, past the units the buffers touch.
    const LEN: usize = 0x10000;

    /// What a device counted of the maintenance: flushes, their bytes, invalidations, their bytes,
    /// imports made in memory that is not coherent.
    #[derive(Clone, Copy, Debug, PartialEq, Eq)]
    struct Tally {
        flushes: u64,
        flush_bytes: u64,
        invalidates: u64,
        invalidate_bytes: u64,
        imports: u64,
    }

    fn tally(device: &Device) -> Tally {
        let counts = device.work_counts();
        Tally {
            flushes: counts.host_flushes,
            flush_bytes: counts.host_flush_bytes,
            invalidates: counts.host_invalidates,
            invalidate_bytes: counts.host_invalidate_bytes,
            imports: counts.noncoherent_imports_made,
        }
    }

    /// `LEN` bytes of private anonymous pages from a multiple of `unit`, holding `counting(LEN, 0)`;
    /// unmapped when dropped, after the buffers over them (declared after it).
    struct Pages {
        map: *mut libc::c_void,
        size: usize,
        base: NonNull<u8>,
    }

    impl Pages {
        fn new(unit: usize) -> Pages {
            let size = LEN + unit;
            // SAFETY: a new private mapping, this object's until dropped.
            let map = unsafe {
                libc::mmap(
                    std::ptr::null_mut(),
                    size,
                    libc::PROT_READ | libc::PROT_WRITE,
                    libc::MAP_PRIVATE | libc::MAP_ANONYMOUS,
                    -1,
                    0,
                )
            };
            assert_ne!(map, libc::MAP_FAILED, "mmap of the test's pages");
            let base = (map as usize).next_multiple_of(unit);
            let pages = Pages {
                map,
                size,
                base: NonNull::new(base as *mut u8).unwrap(),
            };
            pages.write(0, &counting(LEN, 0));
            pages
        }

        fn pointer(&self, offset: usize) -> NonNull<u8> {
            assert!(offset <= LEN);
            // SAFETY: inside the mapping.
            unsafe { NonNull::new_unchecked(self.base.as_ptr().add(offset)) }
        }

        fn write(&self, offset: usize, bytes: &[u8]) {
            assert!(offset + bytes.len() <= LEN);
            // SAFETY: inside the mapping, which no command writes while a test calls this.
            unsafe {
                std::ptr::copy_nonoverlapping(
                    bytes.as_ptr(),
                    self.pointer(offset).as_ptr(),
                    bytes.len(),
                )
            };
        }

        fn read(&self, offset: usize, length: usize) -> Vec<u8> {
            assert!(offset + length <= LEN);
            // SAFETY: inside the mapping, which no command writes while a test calls this.
            unsafe { std::slice::from_raw_parts(self.pointer(offset).as_ptr(), length).to_vec() }
        }

        /// A no-copy buffer over `length` bytes from `offset`.
        fn buffer(&self, device: &Device, offset: usize, length: usize) -> Buffer {
            // SAFETY: the pages outlive the device's buffers, which the tests drop before them.
            unsafe {
                device.new_buffer_with_bytes_no_copy(
                    HostMemory::new(self.pointer(offset)),
                    length as u64,
                    ResourceOptions::default(),
                    None,
                )
            }
            .unwrap()
        }
    }

    impl Drop for Pages {
        fn drop(&mut self) {
            // SAFETY: the mapping `new` made.
            unsafe { libc::munmap(self.map, self.size) };
        }
    }

    /// The device, its queue and the unit its imports are made of, where the device imports host
    /// memory; else `None`, said. Each test makes its device after it forces what it forces.
    struct Rig {
        _scratch: Scratch,
        device: Device,
        queue: CommandQueue,
        unit: usize,
        atom: u64,
    }

    fn rig() -> Option<Rig> {
        let scratch = Scratch::new();
        let device = device(&scratch.0);
        let (page, alignment) = match &device.shared().host_import {
            Ok(importer) => (importer.page, importer.alignment),
            Err(why) => {
                eprintln!(
                    "{TEST}: on {}: the device imports no host memory: {why:?}",
                    device.name()
                );
                return None;
            }
        };
        if alignment > page {
            eprintln!(
                "{TEST}: the import alignment 0x{alignment:x} is above the host page: not run"
            );
            return None;
        }
        let queue = device.new_command_queue().unwrap();
        let atom = device.shared().host_sync.atom();
        let _ = take_host_import_notes();
        Some(Rig {
            _scratch: scratch,
            device,
            queue,
            unit: page.max(alignment) as usize,
            atom,
        })
    }

    /// Commit `command_buffer`, wait for it, and that it ended well.
    fn run(command_buffer: &crate::CommandBuffer) {
        command_buffer.commit().unwrap();
        command_buffer.wait_until_completed_for(WAIT).unwrap();
        assert_eq!(command_buffer.error(), None);
    }

    /// Fill `range` of `buffer` with `value` by a blit: the GPU writes it.
    fn fill(queue: &CommandQueue, buffer: &Buffer, range: std::ops::Range<u64>, value: u8) {
        let command_buffer = queue.command_buffer().unwrap();
        let mut blit = command_buffer.blit_command_encoder().unwrap();
        blit.fill_buffer(buffer, range, value).unwrap();
        blit.end_encoding().unwrap();
        run(&command_buffer);
    }

    /// `buffer`'s first `length` bytes copied by a blit into a buffer of the runtime's own and read
    /// from there: the GPU reads the buffer and writes only the copy.
    fn read(rig: &Rig, buffer: &Buffer, length: usize) -> Vec<u8> {
        let copy = (rig.device)
            .new_buffer(length as u64, ResourceOptions::shared())
            .unwrap();
        let command_buffer = rig.queue.command_buffer().unwrap();
        let mut blit = command_buffer.blit_command_encoder().unwrap();
        blit.copy_from_buffer_to_buffer(buffer, 0, &copy, 0, length as u64)
            .unwrap();
        blit.end_encoding().unwrap();
        run(&command_buffer);
        let at = copy.contents().unwrap();
        // SAFETY: the copy's own bytes, which its completed command buffer wrote.
        unsafe { std::slice::from_raw_parts(at.as_ptr(), length).to_vec() }
    }

    /// The span a no-copy buffer's import is maintained over now, and that it satisfies what a
    /// range of the memory must: whole atoms or the memory's end, covering the buffer's `length`
    /// bytes at `pointer`, inside the memory (the units its pages touch).
    fn span_of(rig: &Rig, buffer: &Buffer, pointer: NonNull<u8>, length: usize) -> Span {
        let pages = buffer
            .host_pages()
            .expect("a no-copy buffer lies in host pages");
        let span = (pages.import().sync_span(rig.atom)).expect("an import that is maintained");
        let address = pointer.as_ptr() as usize;
        let start = address - address % rig.unit;
        let memory_size = ((address + length).next_multiple_of(rig.unit) - start) as u64;
        // The bytes' place in the memory.
        let at = (address - start) as u64;
        assert_eq!(
            pages.offset(),
            at,
            "{TEST}: the bytes' offset in the import"
        );
        assert_eq!(
            span.offset % rig.atom,
            0,
            "{TEST}: the span starts at an atom"
        );
        let end = span.offset + span.size;
        assert!(
            end.is_multiple_of(rig.atom) || end == memory_size,
            "{TEST}: the span ends at an atom or the memory's end: {span:?} of {memory_size}"
        );
        assert!(
            span.offset <= at,
            "{TEST}: it covers the bytes' start: {span:?} at {at}"
        );
        assert!(
            end >= at + length as u64,
            "{TEST}: it covers the bytes' end: {span:?}"
        );
        assert!(
            end <= memory_size,
            "{TEST}: it lies in the memory: {span:?} of {memory_size}"
        );
        span
    }

    /// A buffer the GPU writes is flushed at its command buffer's first submission and, flushed
    /// again and invalidated, at its completion; a buffer it only reads is flushed alone; a
    /// command buffer that uses neither maintains nothing. The bytes the GPU wrote and the CPU
    /// wrote are the ones read.
    #[test]
    fn a_written_buffer_is_flushed_at_submission_and_invalidated_at_completion() {
        let _forced = forced::force(false, None);
        let Some(rig) = rig() else { return };
        let (offset, length) = (0x123, 0x1800);
        let pages = Pages::new(rig.unit);
        let before = tally(&rig.device);
        assert_eq!(before.flushes + before.invalidates, 0);
        assert!(
            !rig.device.shared().host_sync.active(),
            "{TEST}: nothing imported yet"
        );

        let buffer = pages.buffer(&rig.device, offset, length);
        let made = tally(&rig.device);
        assert_eq!(
            made.imports, 1,
            "{TEST}: an import in memory that is not coherent"
        );
        assert_eq!(
            (made.flushes, made.invalidates),
            (0, 0),
            "{TEST}: nothing runs at the import"
        );
        assert!(
            rig.device.shared().host_sync.active(),
            "{TEST}: the maintenance is on"
        );
        assert!(
            take_host_import_notes().is_empty(),
            "{TEST}: the map worked"
        );
        let span = span_of(&rig, &buffer, pages.pointer(offset), length);

        // A command buffer that uses neither: nothing.
        let other = rig
            .device
            .new_buffer(4096, ResourceOptions::shared())
            .unwrap();
        fill(&rig.queue, &other, 0..4096, 0x11);
        assert_eq!(
            tally(&rig.device),
            made,
            "{TEST}: a command buffer without the import"
        );

        // Read only: the CPU's bytes copied out. One flush of the span at the submission, no
        // invalidation.
        let bytes = read(&rig, &buffer, length);
        assert!(
            bytes == counting(LEN, 0)[offset..offset + length],
            "{TEST}: the CPU's bytes"
        );
        let read_only = tally(&rig.device);
        assert_eq!(read_only.flushes, 1, "{TEST}: one flush at the submission");
        assert_eq!(read_only.flush_bytes, span.size, "{TEST}: of the span");
        assert_eq!(
            (read_only.invalidates, read_only.invalidate_bytes),
            (0, 0),
            "{TEST}: a buffer the GPU only read is not invalidated"
        );

        // The CPU writes, the GPU reads what it wrote.
        pages.write(offset + 0x40, &[0xc3; 0x80]);
        let bytes = read(&rig, &buffer, length);
        assert!(
            bytes[0x40..0xc0].iter().all(|&byte| byte == 0xc3),
            "{TEST}: the CPU's write"
        );
        assert_eq!(
            tally(&rig.device).flushes,
            2,
            "{TEST}: one flush for each submission"
        );

        // The GPU writes: one flush at the submission, one (cleaning the range) and one
        // invalidation at the completion.
        let at = tally(&rig.device);
        fill(&rig.queue, &buffer, 0x100..0x200, 0x5a);
        let written = tally(&rig.device);
        assert_eq!(
            written.flushes,
            at.flushes + 2,
            "{TEST}: submission, then before the invalidate"
        );
        assert_eq!(
            written.flush_bytes,
            at.flush_bytes + 2 * span.size,
            "{TEST}"
        );
        assert_eq!(
            written.invalidates,
            at.invalidates + 1,
            "{TEST}: one invalidation"
        );
        assert_eq!(
            written.invalidate_bytes,
            at.invalidate_bytes + span.size,
            "{TEST}"
        );
        let landed = pages.read(offset + 0x100, 0x100);
        assert!(
            landed.iter().all(|&byte| byte == 0x5a),
            "{TEST}: the GPU's fill"
        );
        assert_eq!(
            pages.read(offset + 0x200, 0x10),
            counting(LEN, 0)[offset + 0x200..offset + 0x210],
            "{TEST}: and no byte beside it"
        );
        // Nothing is left planned: the counts stay as the work ended.
        assert_eq!(tally(&rig.device), written);
        drop(buffer);
    }

    /// `flush` mode flushes at the submission and never invalidates; `off` does nothing at all,
    /// not even the flush, and switches the maintenance off, whatever the device imports; the
    /// imports are counted in both, and every byte is as read.
    #[test]
    fn the_modes_flush_only_and_off_maintain_less() {
        for (mode, flushes, invalidates) in [(Mode::Flush, 1, 0), (Mode::Off, 0, 0)] {
            let _forced = forced::force(false, Some(mode));
            let Some(rig) = rig() else { return };
            assert_eq!(rig.device.shared().host_sync.mode(), mode);
            let pages = Pages::new(rig.unit);
            let buffer = pages.buffer(&rig.device, 0, 0x2000);
            assert_eq!(tally(&rig.device).imports, 1, "{TEST}: {mode:?}: counted");
            assert_eq!(
                rig.device.shared().host_sync.active(),
                mode != Mode::Off,
                "{TEST}: {mode:?}: the switch"
            );
            fill(&rig.queue, &buffer, 0..0x100, 0x77);
            let counted = tally(&rig.device);
            assert_eq!(counted.flushes, flushes, "{TEST}: {mode:?}: flushes");
            assert_eq!(
                counted.invalidates, invalidates,
                "{TEST}: {mode:?}: invalidations"
            );
            assert!(
                pages.read(0, 0x100).iter().all(|&byte| byte == 0x77),
                "{TEST}: {mode:?}"
            );
            let bytes = read(&rig, &buffer, 0x2000);
            assert!(
                bytes[0x100..] == counting(LEN, 0)[0x100..0x2000],
                "{TEST}: {mode:?}"
            );
            drop(buffer);
        }
    }

    /// A device that makes no import in memory that is not coherent counts nothing, keeps the
    /// maintenance off, and notes nothing: what a coherent import costs here.
    #[test]
    fn a_coherent_import_counts_nothing() {
        let Some(rig) = rig() else { return };
        let pages = Pages::new(rig.unit);
        let buffer = pages.buffer(&rig.device, 0x123, 0x1800);
        fill(&rig.queue, &buffer, 0x10..0x90, 0x33);
        let bytes = read(&rig, &buffer, 0x1800);
        assert!(bytes[0x10..0x90].iter().all(|&byte| byte == 0x33), "{TEST}");
        assert!(
            !rig.device.shared().host_sync.active(),
            "{TEST}: nothing to maintain"
        );
        assert_eq!(
            tally(&rig.device),
            Tally {
                flushes: 0,
                flush_bytes: 0,
                invalidates: 0,
                invalidate_bytes: 0,
                imports: 0
            },
            "{TEST}"
        );
        assert!(take_host_import_notes().is_empty(), "{TEST}");
        drop(buffer);
    }

    /// An import whose map fails is made all the same, counted as one in memory that is not
    /// coherent, noted at `noncoherent-map`, and used unsynced: no flush, no invalidation, and the
    /// bytes are as read.
    #[test]
    fn a_map_that_fails_leaves_the_import_usable_and_unsynced() {
        let _forced = forced::force(true, None);
        let Some(rig) = rig() else { return };
        let pages = Pages::new(rig.unit);
        let buffer = pages.buffer(&rig.device, 0x40, 0x1000);
        let notes = take_host_import_notes();
        assert!(
            notes.iter().any(|note| {
                note.starts_with("host import unsynced at noncoherent-map: vkMapMemory")
            }),
            "{TEST}: {notes:?}"
        );
        assert_eq!(tally(&rig.device).imports, 1, "{TEST}: counted");
        assert!(
            !rig.device.shared().host_sync.active(),
            "{TEST}: unsynced: nothing to maintain"
        );
        fill(&rig.queue, &buffer, 0..0x80, 0x99);
        assert!(
            pages.read(0x40, 0x80).iter().all(|&byte| byte == 0x99),
            "{TEST}: the fill"
        );
        let bytes = read(&rig, &buffer, 0x1000);
        assert!(
            bytes[0x80..] == counting(LEN, 0)[0x40 + 0x80..0x40 + 0x1000],
            "{TEST}"
        );
        let counted = tally(&rig.device);
        assert_eq!((counted.flushes, counted.invalidates), (0, 0), "{TEST}");
        drop(buffer);
    }

    /// Two imports used by one command buffer, and one used by two encoders of it, are flushed
    /// once each at its submission; the one the GPU wrote is invalidated, the one it read is not.
    #[test]
    fn each_import_a_command_buffer_uses_is_maintained_once() {
        let _forced = forced::force(false, None);
        let Some(rig) = rig() else { return };
        let (first, second) = (Pages::new(rig.unit), Pages::new(rig.unit));
        let source = first.buffer(&rig.device, 0, 0x2000);
        let target = second.buffer(&rig.device, 0x100, 0x1000);
        assert_eq!(tally(&rig.device).imports, 2);
        let command_buffer = rig.queue.command_buffer().unwrap();
        // Two encoders on the source (read both times), one write of the target.
        for round in 0..2u64 {
            let mut blit = command_buffer.blit_command_encoder().unwrap();
            blit.copy_from_buffer_to_buffer(&source, round * 0x100, &target, round * 0x100, 0x100)
                .unwrap();
            blit.end_encoding().unwrap();
        }
        run(&command_buffer);
        let counted = tally(&rig.device);
        let (source_span, target_span) = (
            span_of(&rig, &source, first.pointer(0), 0x2000),
            span_of(&rig, &target, second.pointer(0x100), 0x1000),
        );
        // Flushed once each at the submission; the target cleaned again before its invalidation.
        assert_eq!(counted.flushes, 3, "{TEST}");
        assert_eq!(
            counted.flush_bytes,
            source_span.size + 2 * target_span.size,
            "{TEST}"
        );
        assert_eq!(counted.invalidates, 1, "{TEST}: only the written import");
        assert_eq!(counted.invalidate_bytes, target_span.size, "{TEST}");
        assert!(
            second.read(0x100, 0x200) == counting(LEN, 0)[0..0x200],
            "{TEST}: the copy"
        );
        drop((source, target));
    }

    /// A rebind gives the buffer a new import, also in memory that is not coherent, which the
    /// next command buffer flushes and invalidates in place of the old one's.
    #[test]
    fn a_rebound_buffer_is_maintained_over_its_new_pages() {
        let _forced = forced::force(false, None);
        let Some(rig) = rig() else { return };
        let (old, new) = (Pages::new(rig.unit), Pages::new(rig.unit));
        new.write(0, &[0xab; LEN]);
        let buffer = old.buffer(&rig.device, 0, 0x2000);
        let first = span_of(&rig, &buffer, old.pointer(0), 0x2000);
        fill(&rig.queue, &buffer, 0..0x40, 1);
        assert_eq!(tally(&rig.device).invalidates, 1, "{TEST}");
        // SAFETY: the new pages outlive the buffer, which the test drops before them.
        unsafe {
            buffer
                .rebind_bytes_no_copy(HostMemory::new(new.pointer(0)), None, None)
                .unwrap();
        }
        assert_eq!(
            tally(&rig.device).imports,
            2,
            "{TEST}: the rebind imported again"
        );
        let second = span_of(&rig, &buffer, new.pointer(0), 0x2000);
        assert_ne!(first.memory, second.memory, "{TEST}: another memory");
        let before = tally(&rig.device);
        fill(&rig.queue, &buffer, 0x100..0x140, 2);
        let after = tally(&rig.device);
        assert_eq!(after.flushes, before.flushes + 2, "{TEST}");
        assert_eq!(
            after.flush_bytes,
            before.flush_bytes + 2 * second.size,
            "{TEST}"
        );
        assert_eq!(after.invalidates, before.invalidates + 1, "{TEST}");
        assert!(
            new.read(0x100, 0x40).iter().all(|&byte| byte == 2),
            "{TEST}: the new pages"
        );
        assert!(
            old.read(0x100, 0x40) == counting(LEN, 0)[0x100..0x140],
            "{TEST}: not the old"
        );
        drop(buffer);
    }

    /// A heap over host memory is one import: a buffer placed in it, written by the GPU, has the
    /// heap's whole reach flushed and invalidated.
    #[test]
    fn a_heap_over_host_memory_is_maintained_through_its_buffers() {
        let _forced = forced::force(false, None);
        let Some(rig) = rig() else { return };
        let pages = Pages::new(rig.unit);
        let size = 0x4000u64;
        let descriptor = HeapDescriptor::new(HeapType::Automatic, size, ResourceOptions::shared());
        // SAFETY: the pages outlive the heap, which the test drops before them.
        let heap = unsafe {
            rig.device.new_heap_with_bytes_no_copy(
                HostMemory::new(pages.pointer(0)),
                &descriptor,
                None,
                None,
            )
        };
        let heap = match heap {
            Ok(heap) => heap,
            Err(error) => {
                eprintln!(
                    "{TEST}: on {}: no no-copy heap: {error}: not run",
                    rig.device.name()
                );
                return;
            }
        };
        let buffer = heap.new_buffer(0x1000, ResourceOptions::shared()).unwrap();
        assert_eq!(
            tally(&rig.device).imports,
            1,
            "{TEST}: the heap's own import"
        );
        let before = tally(&rig.device);
        fill(&rig.queue, &buffer, 0..0x100, 0x6d);
        let after = tally(&rig.device);
        assert_eq!(
            after.flushes,
            before.flushes + 2,
            "{TEST}: the heap's span, flushed twice"
        );
        assert_eq!(
            after.invalidates,
            before.invalidates + 1,
            "{TEST}: and invalidated"
        );
        assert!(
            pages.read(0, 0x100).iter().all(|&byte| byte == 0x6d),
            "{TEST}: the fill"
        );
        drop((buffer, heap));
    }

    /// A command buffer whose encoder opened before the device's first import in memory that is
    /// not coherent noted no uses (`Taken::opened`), so it counts every buffer it names as one
    /// the GPU may have written: invalidated though it only read it. One opened after notes its
    /// uses, and invalidates none it only read.
    #[test]
    fn an_encoder_opened_before_the_first_such_import_invalidates_what_it_names() {
        let _forced = forced::force(false, None);
        let Some(rig) = rig() else { return };
        let pages = Pages::new(rig.unit);
        // The encoder opens while the device has no import to maintain.
        let early = rig.queue.command_buffer().unwrap();
        let mut blit = early.blit_command_encoder().unwrap();
        assert!(!rig.device.shared().host_sync.active());
        let buffer = pages.buffer(&rig.device, 0, 0x1000);
        assert!(rig.device.shared().host_sync.active());
        let copy = rig
            .device
            .new_buffer(0x1000, ResourceOptions::shared())
            .unwrap();
        blit.copy_from_buffer_to_buffer(&buffer, 0, &copy, 0, 0x1000)
            .unwrap();
        blit.end_encoding().unwrap();
        run(&early);
        let counted = tally(&rig.device);
        assert_eq!(
            counted.flushes, 2,
            "{TEST}: flushed, and cleaned for the invalidate"
        );
        assert_eq!(
            counted.invalidates, 1,
            "{TEST}: invalidated, though only read"
        );
        // An encoder opened now notes its uses: the same read invalidates nothing more.
        let bytes = read(&rig, &buffer, 0x1000);
        assert!(bytes == counting(LEN, 0)[..0x1000], "{TEST}");
        let after = tally(&rig.device);
        assert_eq!(after.flushes, 3, "{TEST}");
        assert_eq!(
            after.invalidates, 1,
            "{TEST}: a read alone is no invalidation"
        );
        drop((buffer, copy));
    }

    /// A texture over a no-copy buffer lies in the buffer's bytes (here a shadow of them, as
    /// lavapipe cannot alias an image over host pages, merged back after each write): the GPU's
    /// write into it is a write of the buffer, so the buffer's span is flushed and invalidated,
    /// and the CPU reads the texels through the pages.
    #[test]
    fn a_texture_over_an_import_is_maintained_through_its_buffer() {
        use crate::{BlitOption, Origin, Size, TextureDescriptor};
        use metal2vulkan::metal::{PixelFormat, TextureUsage};
        let _forced = forced::force(false, None);
        let Some(rig) = rig() else { return };
        let pages = Pages::new(rig.unit);
        let (side, row) = (16u64, 64u64);
        let buffer = pages.buffer(&rig.device, 0, 0x2000);
        let mut descriptor =
            TextureDescriptor::texture_2d(PixelFormat::RGBA8Unorm, side, side, false);
        descriptor.usage = TextureUsage::SHADER_READ | TextureUsage::SHADER_WRITE;
        descriptor.resource_options = buffer.resource_options();
        let texture = match buffer.new_texture(&descriptor, 0, row) {
            Ok(texture) => texture,
            Err(error) => {
                eprintln!("{TEST}: no texture over host pages: {error}: not run");
                return;
            }
        };
        let texels: Vec<u8> = (0..side * row).map(|byte| (byte % 253) as u8).collect();
        let source = rig
            .device
            .new_buffer(side * row, ResourceOptions::shared())
            .unwrap();
        // SAFETY: the buffer's own bytes, which no command uses yet.
        unsafe {
            std::ptr::copy_nonoverlapping(
                texels.as_ptr(),
                source.contents().unwrap().as_ptr(),
                texels.len(),
            )
        };
        let before = tally(&rig.device);
        let command_buffer = rig.queue.command_buffer().unwrap();
        let mut blit = command_buffer.blit_command_encoder().unwrap();
        blit.copy_from_buffer_to_texture(
            &source,
            0,
            row,
            0,
            Size::new(side, side, 1),
            &texture,
            0,
            0,
            Origin { x: 0, y: 0, z: 0 },
            BlitOption::None,
        )
        .unwrap();
        blit.end_encoding().unwrap();
        run(&command_buffer);
        let after = tally(&rig.device);
        assert!(
            after.flushes > before.flushes,
            "{TEST}: the buffer's span is flushed: {before:?} then {after:?}"
        );
        assert_eq!(
            after.invalidates,
            before.invalidates + 1,
            "{TEST}: and invalidated, as the texture's write is the buffer's: {before:?} then              {after:?}"
        );
        assert!(
            pages.read(0, texels.len()) == texels,
            "{TEST}: the texels, read through the pages"
        );
        drop((texture, source, buffer));
    }

    /// The probe counts the imports it made in memory that is not coherent in its facts line, and
    /// still answers `ok` for each shape that succeeded there.
    #[test]
    fn the_probe_counts_the_imports_it_made_in_non_coherent_memory() {
        let _forced = forced::force(false, None);
        let Some(rig) = rig() else { return };
        let lines = rig.device.probe_host_import();
        let (_, noncoherent) = lines[0]
            .rsplit_once(" noncoherent_imports=")
            .unwrap_or_else(|| panic!("{lines:#?}"));
        let noncoherent: u64 = noncoherent.parse().unwrap();
        let ok = lines[1..]
            .iter()
            .filter(|line| line.contains(": ok"))
            .count();
        assert!(ok >= 4, "{lines:#?}");
        // One import for each shape that made its own, whatever the shape.
        assert!(noncoherent >= 1 && noncoherent as usize <= ok, "{lines:#?}");
        assert_eq!(
            tally(&rig.device).imports,
            noncoherent,
            "{TEST}: {lines:#?}"
        );
    }
}
