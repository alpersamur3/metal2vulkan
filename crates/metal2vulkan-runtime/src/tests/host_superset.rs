//! A no-copy buffer over a range its device's import alignment cannot cover, imported over the
//! range's aligned superset (`host_memory.rs`'s `superset_window`, `buffer_descriptor.rs`'s
//! `import_superset`), on the device the cfg selects.
//!
//! No driver of the machines these tests run on imports at an alignment above the host page, so
//! each test makes its device import at 64 KiB (`forced`), which the driver accepts
//! over its own 4 KiB: the runtime then takes the paths a device of such an alignment takes, and
//! the pages a pointer import is made over are the ones the runtime chose. The tests check:
//! - the superset is imported where every page of it is mapped writable, with the buffer's bytes
//!   at their offset in it: the bytes a blit reads and writes (and no byte beside them), the
//!   import's bytes (`WorkCounts::host_pointer_import_bytes_made`), no note of a refusal;
//! - a released import is made again over the superset (the tests take the driver's pointer
//!   imports to follow their address, as the driver of a phone that needs the superset may);
//! - a superset with a page the process cannot write or does not map is not imported: the call
//!   is refused as it was, with the notes that say which check refused.
//!
//! A device test: compiled only under the validation crate's GPU safety cfgs (`m2v_gpu`,
//! `m2v_cpu_icd`), on Linux, on the kind of device the cfg selects.

use super::host_residency::{counting, device, gpu_fill, read, release, residency, Scratch};
use crate::host_import_note::take_host_import_notes;
use crate::host_memory::forced;
use crate::{Buffer, Device, Error, HostMemory, ResourceOptions};
use metal2vulkan::error::Feature;
use std::ptr::NonNull;

const TEST: &str = "aligned superset";

/// The import alignment the tests' devices are made to have.
const ALIGNMENT: u64 = 64 << 10;

/// The bytes of a test's pages: ten alignments' worth would do, eight are enough.
const LEN: usize = 8 * ALIGNMENT as usize;

/// `LEN` bytes of private anonymous pages from a multiple of [`ALIGNMENT`], holding
/// `counting(LEN, 0)`; unmapped when dropped, after the buffers over them (declared after it).
struct Pages {
    map: *mut libc::c_void,
    base: NonNull<u8>,
}

impl Pages {
    fn new() -> Pages {
        let size = LEN + ALIGNMENT as usize;
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
        let base = (map as usize).next_multiple_of(ALIGNMENT as usize);
        let pages = Pages {
            map,
            base: NonNull::new(base as *mut u8).unwrap(),
        };
        pages.write(0, &counting(LEN, 0));
        pages
    }

    /// The address `offset` bytes in.
    fn pointer(&self, offset: usize) -> NonNull<u8> {
        assert!(offset <= LEN);
        // SAFETY: inside the mapping.
        unsafe { NonNull::new_unchecked(self.base.as_ptr().add(offset)) }
    }

    /// Write `bytes` at `offset`.
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

    /// The bytes from `offset`, `length` of them.
    fn read(&self, offset: usize, length: usize) -> Vec<u8> {
        assert!(offset + length <= LEN);
        // SAFETY: inside the mapping, which no command writes while a test calls this.
        unsafe { std::slice::from_raw_parts(self.pointer(offset).as_ptr(), length).to_vec() }
    }

    /// A no-copy buffer over `length` bytes from `offset`.
    fn buffer(&self, device: &Device, offset: usize, length: usize) -> Result<Buffer, Error> {
        // SAFETY: the pages outlive the device's buffers, which the tests drop before them.
        unsafe {
            device.new_buffer_with_bytes_no_copy(
                HostMemory::new(self.pointer(offset)),
                length as u64,
                ResourceOptions::default(),
                None,
            )
        }
    }
}

impl Drop for Pages {
    fn drop(&mut self) {
        // SAFETY: the mapping `new` made.
        unsafe { libc::munmap(self.map, LEN + ALIGNMENT as usize) };
    }
}

/// The imports by pointer the device made, and their bytes.
fn pointer_imports(device: &Device) -> (u64, u64) {
    let counts = device.work_counts();
    (
        counts.host_pointer_imports_made,
        counts.host_pointer_import_bytes_made,
    )
}

/// Run `test` on a device made to import at [`ALIGNMENT`], where the device imports host memory
/// at all and the host page is below that alignment.
fn on_device(test: impl FnOnce(&Device, &Pages)) {
    let _forced = forced::force(ALIGNMENT, true);
    let pages = Pages::new();
    let scratch = Scratch::new();
    let device = device(&scratch.0);
    let (page, alignment) = match &device.shared().host_import {
        Ok(importer) => (importer.page, importer.alignment),
        Err(why) => {
            eprintln!(
                "{TEST}: on {}: the device imports no host memory: {why:?}",
                device.name()
            );
            return;
        }
    };
    assert_eq!(alignment, ALIGNMENT, "{TEST}: the forced alignment");
    if page >= ALIGNMENT {
        eprintln!("{TEST}: the host page 0x{page:x} is not below the alignment: not run");
        return;
    }
    let _ = take_host_import_notes();
    test(&device, &pages);
}

/// Whether `error` is the refusal of host memory a device cannot import.
fn refused(error: &Error) -> bool {
    matches!(error, Error::Unsupported(unsupported) if unsupported.feature == Feature::HostMemory)
}

/// A buffer over 0x34000 bytes from 0x2000 into a 64-KiB-aligned run of pages: its range ends at
/// 0x36000, short of the unit boundary at 0x40000, so the pages its range touches are no whole
/// number of units. The device imports the units' pages instead, and the buffer reads and writes
/// the bytes it was made over alone.
#[test]
fn a_range_the_alignment_cannot_cover_is_imported_over_its_aligned_superset() {
    on_device(|device, pages| {
        let queue = device.new_command_queue().unwrap();
        let (offset, length) = (0x2000, 0x34000);
        let expected = counting(LEN, 0);

        // A range the alignment covers is imported as ever: its own pages, no note.
        let before = pointer_imports(device);
        let whole = pages.buffer(device, 0, 0x40000).unwrap();
        assert_eq!(pointer_imports(device), (before.0 + 1, before.1 + 0x40000));
        assert!(
            take_host_import_notes().is_empty(),
            "{TEST}: a covered range"
        );
        assert!(
            read(&queue, &whole, 0x40000) == expected[..0x40000],
            "{TEST}"
        );
        drop(whole);

        // This one is not: 0x2000 into a unit, ending at 0x36000 into it.
        let before = pointer_imports(device);
        let buffer = pages.buffer(device, offset, length).unwrap();
        // The units' pages: [0, 0x40000) of the run, 0x40000 bytes.
        assert_eq!(
            pointer_imports(device),
            (before.0 + 1, before.1 + 0x40000),
            "{TEST}: the superset's pages"
        );
        assert!(
            take_host_import_notes().is_empty(),
            "{TEST}: a superset import says nothing"
        );
        assert_eq!(
            buffer.contents(),
            Some(pages.pointer(offset)),
            "{TEST}: contents"
        );
        assert_eq!(buffer.length(), length as u64);
        assert!(
            read(&queue, &buffer, length) == expected[offset..offset + length],
            "{TEST}: the bytes of the range, not of the superset"
        );

        // A GPU write lands in the range, and in no page or byte beside it.
        gpu_fill(&queue, &buffer, 0x100..0x200, 0x5a);
        let mut written = expected.clone();
        written[offset + 0x100..offset + 0x200].fill(0x5a);
        assert!(pages.read(0, LEN) == written, "{TEST}: the GPU's fill");

        // Released while idle and imported again over the superset: the CPU's write made
        // meanwhile is read.
        let counts = device.work_counts();
        let (released, again, resident, _) = residency(&counts);
        release(TEST, device, 1);
        pages.write(offset + 0x300, &[0xee; 64]);
        written[offset + 0x300..offset + 0x340].fill(0xee);
        let bytes = read(&queue, &buffer, length);
        assert!(
            bytes == written[offset..offset + length],
            "{TEST}: after a release"
        );
        let counts = device.work_counts();
        assert_eq!(
            residency(&counts),
            (released + 1, again + 1, resident, 0),
            "{TEST}: released and imported again"
        );
        // The import made again is over the superset's pages too, not the range's own.
        assert_eq!(
            pointer_imports(device),
            (before.0 + 2, before.1 + 2 * 0x40000),
            "{TEST}: imported again over the superset"
        );

        // Its Metal address keeps the pointer's offset in its host page.
        let page = device.shared().host_import.as_ref().unwrap().page;
        assert_eq!(
            buffer.gpu_address().0 % page,
            offset as u64 % page,
            "{TEST}: address"
        );
        drop(buffer);
    });
}

/// A superset with a page this process maps read-only, or does not map, is not imported: the call
/// is refused as it was before the superset existed, and its notes name the checks that refused.
#[test]
fn a_superset_over_a_page_the_process_cannot_write_is_not_imported() {
    on_device(|device, pages| {
        // 0x1d000 bytes from 0x2000: the range's pages end at 0x1f000, the superset's at 0x20000.
        let (offset, length) = (0x2000, 0x1d000);
        let buffer = pages.buffer(device, offset, length).unwrap();
        assert!(
            take_host_import_notes().is_empty(),
            "{TEST}: every page mapped"
        );
        drop(buffer);

        // The first page of the superset, 0x2000 before the range, read-only.
        // SAFETY: a page of the test's mapping.
        let made =
            unsafe { libc::mprotect(pages.pointer(0).as_ptr().cast(), 0x1000, libc::PROT_READ) };
        assert_eq!(made, 0);
        let error = pages.buffer(device, offset, length).err();
        let notes = take_host_import_notes();
        assert!(error.as_ref().is_some_and(refused), "{TEST}: {error:?}");
        for stage in ["unwritable", "superset", "skipped"] {
            let said = format!("host import refused at {stage}:");
            assert!(
                notes.iter().any(|note| note.starts_with(&said)),
                "{TEST}: {notes:?}"
            );
        }
        // The range's own pages are writable still: a range the alignment would cover imports.
        assert!(
            pages.buffer(device, 0x10000, 0x10000).is_ok(),
            "{TEST}: the range's own pages"
        );
        // SAFETY: as above.
        let made = unsafe {
            libc::mprotect(
                pages.pointer(0).as_ptr().cast(),
                0x1000,
                libc::PROT_READ | libc::PROT_WRITE,
            )
        };
        assert_eq!(made, 0);
        assert!(
            pages.buffer(device, offset, length).is_ok(),
            "{TEST}: writable again"
        );
        let _ = take_host_import_notes();

        // The last page of the superset, past the range, not mapped at all. A page this test
        // unmaps may be taken by another test's mapping before the call looks at it (the tests
        // of this crate run on threads of one process), so the test asks again over new pages
        // until a call finds the page unmapped, which one of a few does.
        let mut refused_for_the_hole = false;
        for _ in 0..5 {
            let fresh = Pages::new();
            // SAFETY: a page of the test's mapping, which `Pages` unmaps whole when dropped.
            let made = unsafe { libc::munmap(fresh.pointer(0x1f000).as_ptr().cast(), 0x1000) };
            assert_eq!(made, 0);
            let error = fresh.buffer(device, offset, length).err();
            let notes = take_host_import_notes();
            let said = "host import refused at superset:";
            if error.as_ref().is_some_and(refused) && notes.iter().any(|n| n.starts_with(said)) {
                refused_for_the_hole = true;
                break;
            }
        }
        assert!(
            refused_for_the_hole,
            "{TEST}: a superset with a page not mapped"
        );
    });
}

/// The probe on a device that imports at 64 KiB: the shapes it maps with pages to spare (`P1`,
/// `P2`, `P8`) are imported over their supersets, and a shape whose mapping ends at its range
/// (`P3` to `P7`, which model guest RAM) is imported when its superset happens to be mapped and
/// refused, with the superset's notes, when it is not.
#[test]
fn the_probe_on_a_device_that_imports_at_64_kib_imports_over_supersets_or_says_why() {
    on_device(|device, _| {
        let lines = device.probe_host_import();
        eprintln!("{TEST}: {}\n{}", device.name(), lines.join("\n"));
        assert_eq!(lines.len(), 9, "{lines:#?}");
        assert!(lines[0].contains(" alignment=0x10000 "), "{}", lines[0]);
        for id in ["P1 ", "P2 ", "P8 "] {
            let line = lines.iter().find(|line| line.starts_with(id)).unwrap();
            assert!(line.contains(": ok"), "{lines:#?}");
        }
        for line in &lines[1..] {
            let refusal = line.split_once(": refused: ").map(|(_, notes)| notes);
            if let Some(notes) = refusal {
                assert!(notes.contains("host import refused at superset:"), "{line}");
            }
        }
    });
}
