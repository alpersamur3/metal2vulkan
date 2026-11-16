//! The notes of refused host imports (`host_import_note.rs`): a thread's last four, one line each,
//! made only for a refusal, and printed at most 64 a process.

use crate::host_import_note::{
    note, one_line, printing, take_host_import_notes, Printing, Ring, KEPT, LONGEST, PRINTED,
};
use std::sync::atomic::AtomicU32;

/// This thread's notes, emptied: a test that reads them starts from none, whatever ran on the
/// thread before it.
fn fresh() {
    let _ = take_host_import_notes();
}

#[test]
fn the_ring_keeps_the_last_notes_oldest_first() {
    let mut ring = Ring::new();
    for number in 0..KEPT + 3 {
        ring.push(format!("note {number}"));
    }
    let kept: Vec<String> = (3..KEPT + 3)
        .map(|number| format!("note {number}"))
        .collect();
    assert_eq!(ring.take(), kept);
    assert_eq!(ring.take(), Vec::<String>::new(), "taking empties it");
    ring.push("again".to_owned());
    assert_eq!(ring.take(), ["again"]);
}

#[test]
fn a_note_names_its_stage_and_is_one_line() {
    fresh();
    note("binding", || "window 0x4000\nbuffer 0x1000\r\n".to_owned());
    let notes = take_host_import_notes();
    assert_eq!(
        notes,
        ["host import refused at binding: window 0x4000 buffer 0x1000  "]
    );
    assert!(notes.iter().all(|line| !line.contains(['\n', '\r'])));
    assert!(
        take_host_import_notes().is_empty(),
        "taking drains the notes"
    );
}

#[test]
fn a_thread_keeps_its_last_four_notes_oldest_first() {
    fresh();
    for stage in [
        "window",
        "alignment",
        "skipped",
        "unwritable",
        "buffer",
        "bind",
    ] {
        note(stage, || format!("the {stage} check"));
    }
    let notes = take_host_import_notes();
    assert_eq!(
        notes,
        [
            "host import refused at skipped: the skipped check",
            "host import refused at unwritable: the unwritable check",
            "host import refused at buffer: the buffer check",
            "host import refused at bind: the bind check",
        ]
    );
}

#[test]
fn a_thread_s_notes_are_its_own() {
    fresh();
    note("window", || "this thread's".to_owned());
    std::thread::spawn(|| {
        assert!(take_host_import_notes().is_empty());
        note("bind", || "the other thread's".to_owned());
        assert_eq!(take_host_import_notes().len(), 1);
    })
    .join()
    .unwrap();
    assert_eq!(
        take_host_import_notes(),
        ["host import refused at window: this thread's"]
    );
}

#[test]
fn the_numbers_are_formatted_only_for_a_note() {
    fresh();
    let mut formatted = 0;
    note("allocate", || {
        formatted += 1;
        "once".to_owned()
    });
    assert_eq!(formatted, 1);
    assert_eq!(take_host_import_notes().len(), 1);
    // A check that passes makes no note, so what it would have said is never built: this is the
    // shape of every call site (`if refused { note(..) }`), here against the kernel's answer for
    // a window this process maps. The mapping is of 16 pages and a page is made read-only, not
    // unmapped: a page the test unmapped could be taken by another test's one-page mapping
    // before it asked (`this_process_s_read_only_and_unmapped_pages_are_refused`), and a
    // one-page mapping of this test's could take another's.
    #[cfg(any(target_os = "linux", target_os = "android"))]
    {
        use crate::host_memory::{host_page, refuse_unwritable, ImportWindow};
        use crate::work_counts::Counters;
        let page = host_page().unwrap();
        let len = 16 * page as usize;
        // SAFETY: a fresh private mapping, unmapped below.
        let at = unsafe {
            libc::mmap(
                std::ptr::null_mut(),
                len,
                libc::PROT_READ | libc::PROT_WRITE,
                libc::MAP_PRIVATE | libc::MAP_ANON,
                -1,
                0,
            )
        };
        assert_ne!(at, libc::MAP_FAILED);
        let window = ImportWindow {
            start: at as u64,
            size: page,
            offset: 0,
        };
        let counts = Counters::default();
        assert_eq!(refuse_unwritable(&window, &counts), Ok(()));
        assert!(
            take_host_import_notes().is_empty(),
            "a writable window says nothing"
        );
        // SAFETY: the mapping is this test's.
        assert_eq!(
            unsafe { libc::mprotect(at, page as usize, libc::PROT_READ) },
            0
        );
        assert!(refuse_unwritable(&window, &counts).is_err());
        let notes = take_host_import_notes();
        assert_eq!(notes.len(), 1, "{notes:?}");
        let expected = format!(
            "host import refused at unwritable: window 0x{:x} size 0x{page:x}",
            window.start
        );
        assert!(notes[0].starts_with(&expected), "{notes:?}");
        // SAFETY: the mapping is this test's.
        assert_eq!(unsafe { libc::munmap(at, len) }, 0);
    }
}

#[test]
fn a_note_is_cut_to_its_longest_whole_characters() {
    let short = "a".repeat(LONGEST);
    assert_eq!(one_line(short.clone()), short);
    let long = "é".repeat(LONGEST + 50);
    let cut = one_line(long);
    assert_eq!(cut.chars().count(), LONGEST);
    assert!(cut.ends_with("..."));
    assert!(cut.starts_with("éé"));
    fresh();
    note("allocate", || "x".repeat(10_000));
    let notes = take_host_import_notes();
    assert_eq!(notes[0].chars().count(), LONGEST, "the whole note is cut");
}

#[test]
fn the_process_prints_its_first_64_notes_then_one_line_and_no_more() {
    // 64 of them, as the module says.
    assert_eq!(PRINTED, 64);
    let printed = AtomicU32::new(0);
    let mut lines = 0;
    for _ in 0..PRINTED {
        assert_eq!(printing(&printed, PRINTED), Printing::Line);
        lines += 1;
    }
    assert_eq!(lines, 64);
    assert_eq!(printing(&printed, PRINTED), Printing::Last, "the 65th");
    for _ in 0..1000 {
        assert_eq!(printing(&printed, PRINTED), Printing::Quiet);
    }
    // The counter stops where it is, so it never wraps back to printing.
    let counted = printed.load(std::sync::atomic::Ordering::Relaxed);
    assert_eq!(counted, PRINTED + 1);
    // A cap of none says it once, then nothing.
    let none = AtomicU32::new(0);
    assert_eq!(printing(&none, 0), Printing::Last);
    assert_eq!(printing(&none, 0), Printing::Quiet);
}
