//! The import probe ([`Device::probe_host_import`]): what the device's driver accepts of the
//! shapes of host memory a program hands to `newBufferWithBytesNoCopy:`, answered on the device
//! itself, for a machine nobody can run the program on.
//!
//! The probe goes through the public no-copy call, on mappings it makes itself, so every check a
//! real buffer meets (the import window, the writability of the pages, the buffer's binding, the
//! memory types, the driver's own answer) is the one a real buffer meets, and each refusal comes
//! with the notes that say which check it was (`host_import_note.rs`). It answers a line of the
//! importer's facts, then one line for each shape: `<id> <name>: ok`, `<id> <name>: refused:
//! <notes>` or `<id> <name>: error: <the error>`. It imports no dma-bufs: the shapes are the
//! pointer's.
//!
//! The shapes, each 0x34000 bytes (13 pages of 16 KiB, the length a Mali phone refused) but the
//! last:
//! - `P1` private anonymous pages at a multiple of 64 KiB;
//! - `P2` private anonymous pages, the pointer 0x2000 into such a mapping: not at 16 KiB;
//! - `P3` one shared mapping of a memory file (a `memfd`: guest RAM as QEMU makes it);
//! - `P4` a shared mapping of a memory file made of 13 mappings of 16 KiB each, side by side,
//!   the file's runs in the reverse order: 13 mappings of the process behind one pointer;
//! - `P5` the pages of `P3` mapped a second time, imported while `P3`'s buffer lives: the same
//!   pages twice;
//! - `P6` 0x30000 bytes from 0x4000 into `P3`'s pages, while `P3`'s buffer lives: contained in
//!   an import, not the same bytes;
//! - `P7` the same bytes as `P6` made again after `P3`'s buffer, and so its import, went;
//! - `P8` 0x1717000 bytes (24 MiB) of private anonymous pages.
//!
//! Every buffer is dropped, and then every mapping unmapped, before the probe returns.

use super::device::lock;
use super::host_import_note::take_host_import_notes;
use super::{Buffer, Device, Error, HostMemory, ResourceOptions};
use std::os::fd::{AsRawFd, OwnedFd};
use std::ptr::NonNull;

/// The bytes of a shape but the last: 13 pages of 16 KiB.
const LENGTH: usize = 0x34000;

/// A 16 KiB run of the memory file that `P4` maps.
const RUN: usize = 0x4000;

/// A mapping of this process, unmapped when dropped.
struct Mapping {
    at: NonNull<u8>,
    len: usize,
}

impl Mapping {
    /// A new mapping of `len` bytes, anywhere: `prot`, `flags`, and the file from an offset where
    /// it maps one.
    fn new(
        len: usize,
        (prot, flags): (libc::c_int, libc::c_int),
        file: Option<(&OwnedFd, usize)>,
    ) -> std::io::Result<Mapping> {
        let (fd, offset) = match file {
            Some((fd, offset)) => (fd.as_raw_fd(), offset),
            None => (-1, 0),
        };
        // SAFETY: a new mapping at an address of the system's choosing, which this object unmaps
        // when dropped.
        let at = unsafe {
            libc::mmap(
                std::ptr::null_mut(),
                len,
                prot,
                flags,
                fd,
                offset as libc::off_t,
            )
        };
        match NonNull::new(at.cast()).filter(|_| at != libc::MAP_FAILED) {
            Some(at) => Ok(Mapping { at, len }),
            None => Err(std::io::Error::last_os_error()),
        }
    }

    /// `len` bytes of private anonymous pages, readable and writable.
    fn anonymous(len: usize) -> std::io::Result<Mapping> {
        let flags = libc::MAP_PRIVATE | libc::MAP_ANON;
        Mapping::new(len, (libc::PROT_READ | libc::PROT_WRITE, flags), None)
    }

    /// `len` bytes of the memory file `fd` from `offset`, shared, readable and writable.
    fn shared(fd: &OwnedFd, offset: usize, len: usize) -> std::io::Result<Mapping> {
        let prot = libc::PROT_READ | libc::PROT_WRITE;
        Mapping::new(len, (prot, libc::MAP_SHARED), Some((fd, offset)))
    }

    /// The address `offset` bytes in, where it lies in the mapping.
    fn pointer(&self, offset: usize) -> Option<NonNull<u8>> {
        if offset > self.len {
            return None;
        }
        NonNull::new(self.at.as_ptr().wrapping_add(offset))
    }

    /// The first address in the mapping that is a multiple of `unit`, where `len` bytes from it
    /// are in the mapping too.
    fn aligned(&self, unit: usize, len: usize) -> Option<NonNull<u8>> {
        let at = self.at.as_ptr() as usize;
        let skip = at.checked_next_multiple_of(unit)? - at;
        if skip.checked_add(len)? > self.len {
            return None;
        }
        self.pointer(skip)
    }

    /// The `len` bytes from `at`, which lie in this mapping, replaced by the memory file `fd`
    /// from `offset`, shared: the mapping keeps them, and unmaps them with the rest.
    fn replace_with_file(
        &self,
        at: NonNull<u8>,
        len: usize,
        (fd, offset): (&OwnedFd, usize),
    ) -> std::io::Result<()> {
        let end = (at.as_ptr() as usize).checked_add(len);
        let whole = (self.at.as_ptr() as usize).checked_add(self.len);
        if at < self.at || end.is_none() || end > whole {
            return Err(no_address());
        }
        let prot = libc::PROT_READ | libc::PROT_WRITE;
        // SAFETY: the range lies in this mapping, which holds it only to have it replaced.
        let replaced = unsafe {
            libc::mmap(
                at.as_ptr().cast(),
                len,
                prot,
                libc::MAP_SHARED | libc::MAP_FIXED,
                fd.as_raw_fd(),
                offset as libc::off_t,
            )
        };
        match replaced == libc::MAP_FAILED {
            true => Err(std::io::Error::last_os_error()),
            false => Ok(()),
        }
    }
}

impl Drop for Mapping {
    fn drop(&mut self) {
        // SAFETY: a mapping `new` made, whose buffers the probe dropped first.
        unsafe { libc::munmap(self.at.as_ptr().cast(), self.len) };
    }
}

/// A memory file of `len` bytes (a `memfd` where the system has one), open for reading and
/// writing, which the mappings of the shapes share.
#[cfg(any(target_os = "linux", target_os = "android"))]
fn memory_file(len: usize) -> std::io::Result<OwnedFd> {
    use std::os::fd::FromRawFd;
    // SAFETY: a NUL-terminated name.
    let fd = unsafe { libc::memfd_create(c"m2v-host-probe".as_ptr(), libc::MFD_CLOEXEC) };
    if fd < 0 {
        return Err(std::io::Error::last_os_error());
    }
    // SAFETY: a new descriptor the call returned, owned at once.
    let fd = unsafe { OwnedFd::from_raw_fd(fd) };
    // SAFETY: the descriptor is open, and a file of this size holds no memory of this process.
    if unsafe { libc::ftruncate(fd.as_raw_fd(), len as libc::off_t) } != 0 {
        return Err(std::io::Error::last_os_error());
    }
    Ok(fd)
}

/// A memory file of `len` bytes: an unlinked temporary file, where the system has no `memfd`.
#[cfg(not(any(target_os = "linux", target_os = "android")))]
fn memory_file(len: usize) -> std::io::Result<OwnedFd> {
    let path = std::env::temp_dir().join(format!("m2v-host-probe-{}", std::process::id()));
    let file = std::fs::OpenOptions::new()
        .read(true)
        .write(true)
        .create_new(true)
        .open(&path)?;
    let _ = std::fs::remove_file(&path);
    file.set_len(len as u64)?;
    Ok(OwnedFd::from(file))
}

/// The pointer and the length a shape is imported at, or why it could not be made.
type Made = std::io::Result<(NonNull<u8>, usize)>;

/// Why a shape was not made where the mapping holds no address it needs.
fn no_address() -> std::io::Error {
    std::io::Error::other("the mapping holds no such address")
}

/// `P1`: private anonymous pages, the pointer at a multiple of 64 KiB (or of `unit`, where the
/// device imports at more), with mapped pages enough past the range for a device that imports
/// whole units.
fn aligned_anonymous(held: &mut Vec<Mapping>, unit: usize) -> Made {
    let map = Mapping::anonymous(LENGTH + 2 * unit)?;
    let at = map.aligned(unit, LENGTH + unit).ok_or_else(no_address)?;
    held.push(map);
    Ok((at, LENGTH))
}

/// `P2`: private anonymous pages, the pointer 0x2000 into a multiple of `unit`, with mapped pages
/// enough on both sides for a device that imports whole units.
fn unaligned_anonymous(held: &mut Vec<Mapping>, unit: usize) -> Made {
    let map = Mapping::anonymous(LENGTH + 2 * unit + 0x2000)?;
    let aligned = map
        .aligned(unit, LENGTH + unit + 0x2000)
        .ok_or_else(no_address)?;
    let at = NonNull::new(aligned.as_ptr().wrapping_add(0x2000)).ok_or_else(no_address)?;
    held.push(map);
    Ok((at, LENGTH))
}

/// `P3`: one shared mapping of a memory file, which comes with it for `P5`.
fn one_shared_mapping(held: &mut Vec<Mapping>) -> std::io::Result<((NonNull<u8>, usize), OwnedFd)> {
    let file = memory_file(LENGTH)?;
    let map = Mapping::shared(&file, 0, LENGTH)?;
    let at = map.pointer(0).ok_or_else(no_address)?;
    held.push(map);
    Ok(((at, LENGTH), file))
}

/// `P4`: a shared mapping of a memory file of 13 mappings of [`RUN`] bytes side by side in a span
/// reserved for them, the file's runs in the reverse order, so that the 13 cannot merge.
fn many_shared_mappings(held: &mut Vec<Mapping>) -> Made {
    let file = memory_file(LENGTH)?;
    // The span: reserved with no access, then replaced run by run.
    let span = Mapping::new(
        LENGTH,
        (libc::PROT_NONE, libc::MAP_PRIVATE | libc::MAP_ANON),
        None,
    )?;
    let runs = LENGTH / RUN;
    for run in 0..runs {
        let at = span.pointer(run * RUN).ok_or_else(no_address)?;
        span.replace_with_file(at, RUN, (&file, (runs - 1 - run) * RUN))?;
    }
    let at = span.pointer(0).ok_or_else(no_address)?;
    held.push(span);
    Ok((at, LENGTH))
}

/// `P8`: 24 MiB of private anonymous pages, the pointer at a multiple of `unit`, with mapped
/// pages enough past the range for a device that imports whole units.
fn large_anonymous(held: &mut Vec<Mapping>, unit: usize) -> Made {
    const BYTES: usize = 0x1717000;
    let map = Mapping::anonymous(BYTES + 2 * unit)?;
    let at = map.aligned(unit, BYTES + unit).ok_or_else(no_address)?;
    held.push(map);
    Ok((at, BYTES))
}

/// The notes of a refusal, as one text.
fn told(notes: Vec<String>) -> String {
    match notes.is_empty() {
        true => "(no note was made)".to_owned(),
        false => notes.join(" | "),
    }
}

impl Device {
    /// What this device's driver accepts of the shapes of host memory a program hands to
    /// [`Device::new_buffer_with_bytes_no_copy`], as lines of text, for a machine the program
    /// cannot be run on.
    ///
    /// The first line is the importer's facts, `facts: page=0x.. alignment=0x.. dma_buf=..
    /// types=[index:flags ...] noncoherent_imports=N`: the host page, the driver's import
    /// alignment, whether the device imports dma-bufs, the property flags of each of its memory
    /// types, and how many of the imports the probe made were in a type that is not coherent
    /// (`WorkCounts::noncoherent_imports_made`, `host_sync.rs`: 0 where every import took a
    /// coherent type; an `ok` line is also one that succeeded in such a type). Then one line for
    /// each of eight shapes of memory the probe maps itself, `P1` to `P8`: private and shared
    /// pages, one mapping and thirteen, the same pages twice, bytes inside an import, bytes
    /// imported again after the import went, 24 MiB. A line is `<id> <name>: ok`; or `<id> <name>:
    /// refused: <notes>` where the call was refused as memory the device cannot import
    /// (`Unsupported`), the notes being what [`take_host_import_notes`] returned for it, joined by
    /// ` | `, which say which check refused; or `<id> <name>: error: <the error>`. The probe
    /// imports pointers alone, no dma-buf, through the same call and checks a program's buffer
    /// meets. Where the device imports no host memory it answers one line saying why.
    ///
    /// Every buffer it makes is dropped, then every mapping unmapped, before it returns. It never
    /// panics and leaves the device as it found it, but for the notes of its refusals (also
    /// printed, as every note is) and the work counts of its imports. Unix only: it maps memory
    /// with `mmap`.
    ///
    /// [`take_host_import_notes`]: crate::take_host_import_notes
    pub fn probe_host_import(&self) -> Vec<String> {
        let shared = self.shared();
        let importer = match &shared.host_import {
            Ok(importer) => importer,
            Err(why) => return vec![format!("facts: the device imports no host memory: {why:?}")],
        };
        let types: Vec<String> = (lock(&shared.memory).type_flag_bits().iter().enumerate())
            .map(|(index, flags)| format!("{index}:0x{flags:x}"))
            .collect();
        let mut lines = vec![format!(
            "facts: page=0x{:x} alignment=0x{:x} dma_buf={} types=[{}]",
            importer.page,
            importer.alignment,
            importer.imports_dma_bufs(),
            types.join(" ")
        )];
        let noncoherent_before = self.work_counts().noncoherent_imports_made;
        // What the shapes hold: the buffers, dropped first, and the mappings.
        let mut buffers: Vec<Buffer> = Vec::new();
        let mut held: Vec<Mapping> = Vec::new();
        let mut shape = |id: &str, name: &str, made: Made| -> Option<Buffer> {
            let (at, length) = match made {
                Ok(made) => made,
                Err(error) => {
                    lines.push(format!(
                        "{id} {name}: error: the shape was not made: {error}"
                    ));
                    return None;
                }
            };
            let _ = take_host_import_notes();
            // SAFETY: the pages are the probe's own, mapped readable and writable, and stay so
            // until every buffer is dropped, which comes before any mapping is unmapped; nothing
            // else uses them.
            let buffer = unsafe {
                self.new_buffer_with_bytes_no_copy(
                    HostMemory::new(at),
                    length as u64,
                    ResourceOptions::shared(),
                    None,
                )
            };
            match buffer {
                Ok(buffer) => {
                    lines.push(format!("{id} {name}: ok"));
                    Some(buffer)
                }
                Err(Error::Unsupported(_)) => {
                    let notes = told(take_host_import_notes());
                    lines.push(format!("{id} {name}: refused: {notes}"));
                    None
                }
                Err(error) => {
                    let notes = take_host_import_notes();
                    let notes = match notes.is_empty() {
                        true => String::new(),
                        false => format!(" ({})", told(notes)),
                    };
                    lines.push(format!("{id} {name}: error: {error}{notes}"));
                    None
                }
            }
        };

        // 64 KiB, or the alignment where the device imports at more: the pages a shape has past
        // its range, which a device that imports whole units needs.
        let unit = usize::try_from(importer.alignment.max(importer.page))
            .unwrap_or(64 << 10)
            .max(64 << 10);
        let made = aligned_anonymous(&mut held, unit);
        buffers.extend(shape("P1", "anonymous private, 64 KiB aligned", made));
        let made = unaligned_anonymous(&mut held, unit);
        buffers.extend(shape(
            "P2",
            "anonymous private, pointer 0x2000 into a mapping",
            made,
        ));

        // `P3`'s pages and file, for `P5` to `P7`, where the shape was made.
        let (made, file) = match one_shared_mapping(&mut held) {
            Ok((made, file)) => (Ok(made), Some(file)),
            Err(error) => (Err(error), None),
        };
        let shared_at = made.as_ref().ok().map(|&(at, _)| at);
        let mut one = shape("P3", "one memfd mapping, shared", made);

        let made = many_shared_mappings(&mut held);
        buffers.extend(shape(
            "P4",
            "13 memfd mappings of 16 KiB, reversed runs",
            made,
        ));

        // The same pages a second time, while `P3`'s import lives.
        let made = match &file {
            Some(file) => Mapping::shared(file, 0, LENGTH).and_then(|second| {
                let at = second.pointer(0).ok_or_else(no_address)?;
                held.push(second);
                Ok((at, LENGTH))
            }),
            None => Err(no_address()),
        };
        drop(shape(
            "P5",
            "P3's pages mapped again, P3's buffer alive",
            made,
        ));

        // Bytes inside `P3`'s import, then the same bytes after it went.
        let inside = |at: Option<NonNull<u8>>| -> Made {
            let at = at.ok_or_else(no_address)?;
            let at = NonNull::new(at.as_ptr().wrapping_add(0x4000)).ok_or_else(no_address)?;
            Ok((at, 0x30000))
        };
        drop(shape(
            "P6",
            "P3+0x4000 for 0x30000, P3's buffer alive",
            inside(shared_at),
        ));
        drop(one.take());
        let name = "P3+0x4000 for 0x30000 again after P3's buffer went";
        drop(shape("P7", name, inside(shared_at)));

        let made = large_anonymous(&mut held, unit);
        buffers.extend(shape("P8", "anonymous private, 24 MiB", made));

        // The imports the shapes made in memory that is not coherent, counted while they live.
        let noncoherent = self.work_counts().noncoherent_imports_made - noncoherent_before;
        lines[0].push_str(&format!(" noncoherent_imports={noncoherent}"));
        // Every buffer first, then every mapping.
        drop(buffers);
        drop(held);
        drop(file);
        lines
    }
}
