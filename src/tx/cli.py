"""tx — ad-hoc search over Claude Code session transcripts.

Seven traps make naive searching of this corpus silently wrong, and every one
of them produces a confident empty result rather than an error:

  1. Transcripts live at <project>/*.jsonl AND <project>/<uuid>/subagents/*.jsonl.
     A non-recursive glob finds 4 files where the tree holds 340.
  2. Project directory names begin with '-', so shell globs and `find` parse
     them as FLAGS, match nothing, and exit 0. os.walk has no such problem.
  3. Not every `user` record is the operator. Subagent dispatch prompts
     (isSidechain), task notifications, compaction summaries and slash-command
     expansions are all `user` records. Counting them as operator prose
     inflates every result.
  4. Line numbers must match jq's input_line_number so citations are stable:
     one record per line, numbered from 1.
  5. A `user` record's message.content is sometimes a list of blocks and
     sometimes a bare string. Handling only the list form drops whole turns.
  6. Files the operator ATTACHES are not part of their message. They are
     separate top-level `attachment` records whose body hangs at
     .attachment.content.file.content, linked to the message only by
     parentUuid. A selector reading just `user` finds the @-mention path and
     none of the file's words. `say` therefore covers both by default;
     `typed` and `attached` are the narrow halves.
  7. A message the operator types mid-turn is a `queue-operation` record with
     its text at the top level, sometimes with no `user` record at all. A
     selector reading only `user` reports it as never uttered.

Citations print as <transcript>:<line>; `tx cite` reads one back.
"""

from __future__ import annotations

import argparse
import io
import signal
import json
import os
import re
import subprocess
import sys
from collections import Counter
from concurrent.futures import ThreadPoolExecutor

# orjson ships no free-threaded wheel. The source build refuses a
# free-threaded target unless ORJSON_BUILD_FREETHREADED is set, which the
# shebang does, and needs rustc >= 1.95.
import orjson

_loads = orjson.loads

ROOT = os.environ.get("TX_ROOT", os.path.expanduser("~/.claude/projects"))
WIDTH = int(os.environ.get("TX_WIDTH", "130"))
LIMIT = int(os.environ.get("TX_LIMIT", "40"))


def _default_jobs() -> int:
    """Performance cores, not every logical core.

    This work is compute-bound on bytes scanning, so scheduling it onto
    Apple silicon's efficiency cores makes the whole run wait on the slowest
    thread. Measured on 8P+4E: j=8 is 0.36s, j=12 is 0.39s, j=16 is 0.43s.
    """
    try:
        n = int(subprocess.run(
            ["sysctl", "-n", "hw.perflevel0.logicalcpu"],
            capture_output=True, text=True, timeout=1,
        ).stdout.strip())
        if n > 0:
            return n
    except Exception:
        pass
    return os.cpu_count() or 4


JOBS = int(os.environ.get("TX_JOBS", 0)) or _default_jobs()

# `user` records the operator did not type.
INJECTED = re.compile(
    r"<task-notification>|SYSTEM NOTIFICATION - NOT USER INPUT"
    r"|This session is being continued from a previous conversation"
    r"|<fork-boilerplate>|<command-name>|<local-command-stdout>"
    r"|Caveat: The messages below were generated"
)

_WS = re.compile(r"\s+")
_CASE_CLASS = re.compile(r"\[([A-Za-z])([A-Za-z])\]")
_META = set(".^$*+?{}[]()\\")


# Raw records read per turn of requested context; bookkeeping records
# outnumber turns, so a literal window finds nothing.
SPREAD = 12

SUBAGENT_DIR = "subagents"


def here_session() -> str | None:
    """The conversation this process is running inside, if any."""
    return os.environ.get("CLAUDE_CODE_SESSION_ID") or None


def discover(root: str = ROOT, session: str | None = None,
             subagents: bool = False):
    """Every transcript, recursively. Immune to leading-dash directory names.

    A session's own transcript is <uuid>.jsonl and the agents it dispatched are
    <uuid>/subagents/*.jsonl, named for themselves. `session` therefore matches
    the basename OR any directory on the path, so one id selects a conversation
    and the agents it spawned.

    Subagent transcripts are excluded unless `subagents`.
    """
    for dirpath, _dirs, names in os.walk(root):
        parts = dirpath.split(os.sep)
        if not subagents and SUBAGENT_DIR in parts:
            continue
        for n in names:
            if not n.endswith(".jsonl"):
                continue
            if session and session not in n and not any(session in p for p in parts):
                continue
            yield os.path.join(dirpath, n)


# Every row the tool produces, before presentation. Five fields, always in
# this order, so one formatter serves every subcommand.
ROW = "%s\t%d\t%s\t%s\t%s"


def field(s: str) -> str:
    """A value safe to put in a column: no tab, no newline.

    Only the last column holds arbitrary text; the rest are flattened so a tab
    in a filename or tool name shifts no column.
    """
    return _WS.sub(" ", str(s)).strip()


def parse_row(line: str):
    """(transcript, line, time, role, text) from a ROW, or None.

    maxsplit=4 leaves tabs inside the text field.
    """
    parts = line.split("\t", 4)
    if len(parts) != 5 or not parts[1].isdigit():
        return None
    return parts[0], int(parts[1]), parts[2], parts[3], parts[4]


ELIDE = "\x00elide"


MACHINE = "plain"

FIELDS = ("transcript", "line", "time", "role", "text")

# A rendered tool call: `[Name] {json}`.
_TOOL_TEXT = re.compile(r"^\[([A-Za-z_][\w.-]*)\]\s+(\{.*\})$", re.S)


def unwrap_tool(rec: dict) -> dict:
    """Replace a tool call's rendered text with `tool` and a nested `input`.

    Its text is already JSON; as a string the consumer would decode twice.
    """
    m = _TOOL_TEXT.match(rec.get("text") or "")
    if not m:
        return rec
    try:
        args = json.loads(m.group(2))
    except ValueError:
        return rec
    out = {k: v for k, v in rec.items() if k != "text"}
    out["tool"] = m.group(1)
    out["input"] = args
    return out

def toon_table(rows, name: str = "records") -> str:
    """A uniform array of objects in TOON's tabular form.

    Quoting is the reference implementation's: a value needs quoting for `[`,
    `]`, `{`, `}`, `:` and `|` as well as for the delimiter, and a
    numeric-looking string is emitted bare.
    """
    import toon  # imported here so the other formats do not load it

    return toon.encode({name: rows})


def present(rows, out=sys.stdout) -> None:
    """Group by transcript, name it once, then fixed columns.

    A citation stays reconstructible as <transcript>:<line>, and `cite` takes
    the short id. `--tsv` and `--json` skip grouping: one self-contained record
    per line.
    """
    if MACHINE == "tsv":
        for line in rows:
            if line != ELIDE and parse_row(line):
                print(line, file=out)
        return
    if MACHINE in ("json", "toon"):
        records = [dict(zip(FIELDS, r))
                   for r in (parse_row(x) for x in rows if x != ELIDE) if r]
        if MACHINE == "json":
            for rec in records:
                print(orjson.dumps(unwrap_tool(rec)).decode(), file=out)
        elif records:
            # TOON's table wants one scalar per cell, so a tool call stays the
            # rendered string here; --json is where it becomes an object.
            print(toon_table(records), file=out)
        return

    current = None
    for line in rows:
        if line == ELIDE:
            continue
        row = parse_row(line)
        if row is None:
            print(line, file=out)
            continue
        name, lineno, ts, role, text = row
        if name != current:
            if current is not None:
                print(file=out)
            print("# %s" % name, file=out)
            current = name
        print("%-6d %s  %-5s %s" % (lineno, ts, role, text), file=out)


def emit(lines, limit: int, refine: str) -> None:
    """Print at most `limit` lines, then the count held back and the flags
    that narrow or lift the cap."""
    lines = list(lines)
    # Machine output is never elided.
    if limit <= 0 or len(lines) <= limit or MACHINE != "plain":
        present(lines)
        if MACHINE != "plain" and limit > 0 and len(lines) > limit:
            print("tx: %d rows (cap %d not applied to machine output)"
                  % (len(lines), limit), file=sys.stderr)
        return

    # Elided in the MIDDLE: rows are in transcript order, so a tail cut would
    # drop the last records and read as a complete answer.
    head = (limit + 1) // 2
    tail = limit - head
    omitted = len(lines) - limit
    kept = lines[:head] + [ELIDE] + (lines[-tail:] if tail else [])

    # The marker sits at the cut, not only in the footer.
    buf = io.StringIO()
    present(kept, buf)
    text = buf.getvalue()
    parts = text.split("\n")
    # present() drops the sentinel, so re-insert the marker at the boundary.
    shown_head = len([r for r in lines[:head] if parse_row(r)])
    body, seen = [], 0
    for line in parts:
        body.append(line)
        if line and line[0].isdigit():
            seen += 1
            if seen == shown_head:
                body.append("... %d lines omitted ..." % omitted)
    print("\n".join(body).rstrip())

    # Which transcripts the rest are in is the most useful way to narrow, so
    # name them with counts rather than only giving a total.
    per = Counter()
    for line in lines:
        row = parse_row(line)
        if row:
            per[row[0].removesuffix(".jsonl")[:8]] += 1
    spread = ""
    if len(per) > 1:
        top = ", ".join("%s (%d)" % (n, c) for n, c in per.most_common(4))
        more = " +%d more" % (len(per) - 4) if len(per) > 4 else ""
        spread = "  across %d transcripts: %s%s" % (len(per), top, more)

    print("... %d of %d lines omitted from the middle.%s" % (omitted, len(lines), spread))
    print("    narrow: %s   all: -n 0" % refine)


def dedupe_echo(lines, window: int = 60):
    """Drop the second copy of a message the transcript records twice.

    A message typed mid-turn is written when queued and again when delivered,
    a few records apart, with identical text. The earlier line wins.

    Bounded by `window`, so a phrase genuinely repeated later still shows twice.
    """
    seen: dict[tuple[str, str], int] = {}
    out = []
    for line in lines:
        row = parse_row(line)
        if row is None:
            out.append(line)
            continue
        name, n, _ts, _role, text = row
        key = (name, text)
        prev = seen.get(key)
        if prev is not None and n - prev <= window:
            continue
        seen[key] = n
        out.append(line)
    return out


def loc_key(line: str):
    """Sort key putting a file's hits in transcript order.

    Line numbers are decimal, so string order puts 6885 after 31119.
    """
    row = parse_row(line)
    return (row[0], row[1], line) if row else (line, 0, line)


def prefilter(pattern: str):
    r"""Lowercased literals, at least one of which must appear in any line
    whose decoded text can match. None when the pattern is too rich for that.

    The speed of this tool comes from skipping json parsing for text that
    cannot match, so this test MUST NOT have false negatives. It is therefore
    built only from patterns that are a plain alternation of literals --
    `[Cc]addy` counts, since matching is case-insensitive anyway and the class
    collapses to a letter. Anything with real metacharacters returns None and
    everything gets parsed: slower, never wrong.

    Deliberately NOT a compiled regex. `re` with IGNORECASE on bytes cannot
    use a memchr-style scan and costs 2.5s over this corpus, where
    `bytes.lower()` plus `in` does the identical job in 0.57s (measured,
    0.91 GB / 340 files). JSON escaping never rewrites plain ASCII
    alphanumerics, so a literal present in the decoded text is present in
    the raw line too, and `bytes.lower()` is length-preserving, so offsets
    into the lowered copy still address the same lines.
    """
    simplified = _CASE_CLASS.sub(
        lambda m: m.group(1) if m.group(1).lower() == m.group(2).lower() else m.group(0),
        pattern,
    )
    lits = []
    for b in simplified.split("|"):
        if not b or any(c in _META for c in b):
            return None
        lits.append(b.lower().encode())
    return lits


def body_message(rec) -> str | None:
    """text+thinking of a user/assistant record; content may be str or list."""
    msg = rec.get("message")
    if not isinstance(msg, dict):
        return None
    c = msg.get("content")
    if c is None:
        return None
    if isinstance(c, str):
        return c
    if isinstance(c, list):
        out = []
        for b in c:
            if not isinstance(b, dict):
                continue
            t = b.get("type")
            if t == "text":
                out.append(b.get("text") or "")
            elif t == "thinking":
                out.append(b.get("thinking") or "")
        return " ".join(out)
    return json.dumps(c)


def body_queued(rec) -> str | None:
    """Operator prose typed while the assistant was working.

    A message sent mid-turn can exist ONLY here, text at the top level rather
    than under `message`, with no matching `user` record.

    Each is written twice, `enqueue` then `remove`; only the enqueue is taken.
    Harness events arrive the same way and the caller's INJECTED test drops
    them.
    """
    if rec.get("operation") != "enqueue":
        return None
    c = rec.get("content")
    return c if isinstance(c, str) else None


def body_attachment(rec) -> str:
    """The attached file's own text, which sits two levels down."""
    c = (rec.get("attachment") or {}).get("content")
    if isinstance(c, dict):
        f = c.get("file")
        if isinstance(f, dict) and isinstance(f.get("content"), str):
            return f["content"]
    return c if isinstance(c, str) else json.dumps(c)


def is_operator_attachment(rec) -> bool:
    a = rec.get("attachment")
    return (
        isinstance(a, dict)
        and a.get("type") == "file"
        and not rec.get("isSidechain", False)
    )


def work_items(files):
    """One task per file.

    Splitting large files into line-aligned chunks is the right shape in a
    language whose slices borrow -- shifter's Rust miner does exactly that
    over an mmap, and its comment names the reason: with files as the unit,
    the largest transcript alone sets the critical path. It does not transfer
    here. `blob[start:end]` COPIES, so splitting the 98 MB transcript eight
    ways allocates another 98 MB, and the copy costs more than the imbalance
    it removes: measured 0.379s whole-file against 0.440s chunked, and
    single-threaded 1.83s against 1.97s. Revisit with a zero-copy path
    (bounded find over one shared buffer), not with slices.
    """
    for path in files:
        yield (path, None, 0, 0, 1)


def candidate_lines(blob: bytes, pre):
    """Yield (lineno, raw_line) for every line that could possibly match.

    With a literal prefilter this walks MATCH OFFSETS rather than lines.
    Finding the ~9k hits by scanning for them costs measurably less than
    splitting 324k lines and testing each one (1.41x over this corpus), and
    the line numbers stay exact: newlines are counted between consecutive
    hits, so the counting is one left-to-right pass over the blob in total
    rather than one scan per hit.
    """
    if pre is None:
        for i, raw in enumerate(blob.split(b"\n"), 1):
            if raw:
                yield i, raw
        return

    low = blob.lower()
    offsets = set()
    for lit in pre:
        i = low.find(lit)
        while i != -1:
            offsets.add(i)
            i = low.find(lit, i + 1)
    if not offsets:
        return

    prev = 0
    line = 1
    end = -1
    for off in sorted(offsets):
        if off <= end:
            continue  # another hit inside the line already yielded
        line += blob.count(b"\n", prev, off)
        prev = off
        start = blob.rfind(b"\n", 0, off) + 1
        end = blob.find(b"\n", off)
        if end == -1:
            end = len(blob)
        yield line, blob[start:end]


def cut(text: str, width: int) -> str:
    """Collapse runs of whitespace, then truncate -- so a hit is not spent on
    the indentation of the file it was found in."""
    return _WS.sub(" ", text).strip()[:width]


_CFG: dict = {}


def _init(mode, pattern, width, lo=0, hi=0, tslen=10, since="", until=""):
    _CFG["mode"] = mode
    _CFG["re"] = re.compile(pattern, re.I)
    _CFG["pre"] = prefilter(pattern)
    _CFG["width"] = width
    _CFG["lo"] = lo
    _CFG["hi"] = hi
    _CFG["tslen"] = tslen
    _CFG["since"] = since
    _CFG["until"] = until


def stamp(rec, tslen: int) -> str:
    """Record time, to the day at 10 and to the second at 19."""
    return (rec.get("timestamp") or "")[:tslen].replace("T", " ")


def full_stamp(rec) -> str:
    """`YYYY-MM-DD HH:MM:SS`, the form time bounds are compared against."""
    return (rec.get("timestamp") or "")[:19].replace("T", " ")


def time_bound(text: str, end: bool) -> str:
    """A `--since`/`--until` argument widened to a full timestamp.

    Takes what the output prints: a date, or a date and a time. A bare date is
    the whole day. Transcript times are UTC.
    """
    t = text.strip().replace("T", " ")
    if not re.match(r"^\d{4}-\d{2}-\d{2}( \d{2}(:\d{2}(:\d{2})?)?)?$", t):
        raise ValueError("expected YYYY-MM-DD[ HH[:MM[:SS]]], got %r" % text)
    pad = ("23:59:59" if end else "00:00:00")
    if len(t) == 10:
        return "%s %s" % (t, pad)
    return (t + " " + pad[len(t) - 11:])[:19] if len(t) < 19 else t


def _scan(item):
    mode, rx, pre, width = _CFG["mode"], _CFG["re"], _CFG["pre"], _CFG["width"]
    lo, hi, tslen = _CFG["lo"], _CFG["hi"], _CFG["tslen"]
    since, until = _CFG["since"], _CFG["until"]
    path, blob, start, end, first_line = item
    base = os.path.basename(path)
    hits = []
    if blob is None:
        try:
            with open(path, "rb") as fh:
                blob = fh.read()
        except OSError:
            return hits
    elif (start, end) != (0, len(blob)):
        blob = blob[start:end]
    for offset_line, raw in candidate_lines(blob, pre):
        lineno = first_line + offset_line - 1
        if lo and lineno < lo:
            continue
        if hi and lineno > hi:
            continue
        try:
            rec = _loads(raw)
        except Exception:
            continue
        if not isinstance(rec, dict):
            continue
        if since or until:
            # ISO-8601 sorts lexicographically, so a string compare is the
            # whole of the range test.
            when_ts = full_stamp(rec)
            if since and when_ts < since:
                continue
            if until and when_ts > until:
                continue
        rtype = rec.get("type")
        sidechain = rec.get("isSidechain", False)
        src = ""

        if mode in ("say", "typed", "when"):
            if rtype == "user" and not sidechain:
                text = body_message(rec)
                if text is None or INJECTED.search(text):
                    continue
            elif rtype == "queue-operation":
                # Unmarked: the operator typed it, and whether the harness had
                # to queue it is not a distinction they make.
                text = body_queued(rec)
                if text is None or INJECTED.search(text):
                    continue
            elif mode != "typed" and rtype == "attachment" and is_operator_attachment(rec):
                text = body_attachment(rec)
                src = "[%s] " % os.path.basename(
                    (rec.get("attachment") or {}).get("filename") or "?"
                )
            else:
                continue
        elif mode == "attached":
            if rtype != "attachment" or not is_operator_attachment(rec):
                continue
            text = body_attachment(rec)
            src = "[%s] " % os.path.basename(
                (rec.get("attachment") or {}).get("filename") or "?"
            )
        elif mode == "said":
            if rtype != "assistant":
                continue
            text = body_message(rec)
            if text is None:
                continue
        elif mode in ("ran", "used"):
            if rtype != "assistant":
                continue
            msg = rec.get("message")
            blocks = msg.get("content") if isinstance(msg, dict) else None
            if not isinstance(blocks, list):
                continue
            ts = stamp(rec, tslen)
            for b in blocks:
                if not isinstance(b, dict) or b.get("type") != "tool_use":
                    continue
                if mode == "ran":
                    if b.get("name") != "Bash":
                        continue
                    cand = (b.get("input") or {}).get("command") or ""
                    label = ""
                else:
                    if not rx.search(b.get("name") or ""):
                        continue
                    cand = json.dumps(b.get("input") or {})
                    label = (b.get("name") or "") + "\t"
                if mode == "ran" and not rx.search(cand):
                    continue
                role = (b.get("name") or "tool").lower() if mode == "used" else "bash"
                hits.append(ROW % (field(base), lineno, ts, field(role),
                                   cut(cand, width)))
            continue
        else:
            continue

        if not rx.search(text):
            continue
        ts = stamp(rec, tslen)
        if mode == "when":
            hits.append(ts)
        else:
            role = "asst" if mode == "said" else ("file" if src else "user")
            hits.append(ROW % (field(base), lineno, ts, field(role),
                               field(src) + " " + cut(text, width) if src
                               else cut(text, width)))
    return hits


def run(mode, pattern, width, jobs, root, session=None, lo=0, hi=0, tslen=10,
        subagents=False, since="", until=""):
    files = list(discover(root, session, subagents))
    if not files:
        where = f"{root} matching {session!r}" if session else root
        print(f"tx: no transcripts under {where}", file=sys.stderr)
        return []
    out = []
    if jobs <= 1:
        _init(mode, pattern, width, lo, hi, tslen, since, until)
        for item in work_items(files):
            out.extend(_scan(item))
        return out

    # Longest-processing-time-first, over CHUNKS rather than files. Sizes span
    # 144x (98 MB against a 684 KB median), so equal-COUNT chunks in directory
    # order leave whichever thread drew the big one running alone at the end;
    # ordering biggest-first lets the pool self-balance, and splitting the big
    # ones stops any single file from setting the critical path.
    files.sort(key=lambda p: os.path.getsize(p), reverse=True)

    # Threads, not processes: the shebang pins free-threaded 3.14, so there is
    # no GIL to serialize the bytes scanning, and no fork cost, no pickling of
    # results, and no second copy of a file's blob.
    if hasattr(sys, "_is_gil_enabled") and sys._is_gil_enabled():
        print(
            "tx: warning: the GIL is enabled, so threads will not scale. "
            "Expected `uv run --python 3.14t`.",
            file=sys.stderr,
        )
    with ThreadPoolExecutor(
        max_workers=jobs, initializer=_init,
        initargs=(mode, pattern, width, lo, hi, tslen, since, until),
    ) as ex:
        for chunk in ex.map(_scan, work_items(files), chunksize=1):
            out.extend(chunk)
    return out


def citations_from_stdin():
    """FILE:LINE pairs out of whatever was piped in.

    Accepts this tool's own --tsv rows and its grouped output, so a search can
    be narrowed by eye and then read in full without retyping citations.
    """
    out, current = [], None
    for line in sys.stdin:
        line = line.rstrip("\n")
        if line.startswith("# ") and line.endswith(".jsonl"):
            current = line[2:]
            continue
        row = parse_row(line)
        if row:
            out.append("%s:%d" % (row[0], row[1]))
            continue
        m = re.match(r"^(\S+\.jsonl)[:-](\d+)\b", line)
        if m:
            out.append("%s:%s" % (m.group(1), m.group(2)))
            continue
        m = re.match(r"^\s*(\d+)\s", line)
        if m and current:
            out.append("%s:%s" % (current, m.group(1)))
    return out


def cmd_cite(target, root, before=0, after=0, width=WIDTH, subagents=True):
    fname, _, lineno = target.rpartition(":")
    if not fname or not lineno.isdigit():
        print("usage: tx cite FILE:LINE (or - to read citations from stdin)",
              file=sys.stderr)
        return 2
    # Short ids are what the grouped header and the footer print, so accept a
    # prefix as well as the full basename.
    cands = [p for p in discover(root, None, subagents)
             if os.path.basename(p) == fname
             or os.path.basename(p).startswith(fname)]
    path = cands[0] if len(cands) == 1 else None
    if path is None:
        print("no single transcript matching %s (%d candidates)"
              % (fname, len(cands)), file=sys.stderr)
        return 2
    want = int(lineno)
    with open(path, "rb") as fh:
        for i, raw in enumerate(fh, 1):
            if i == want:
                break
        else:
            print(f"{fname} has fewer than {want} lines", file=sys.stderr)
            return 2
    # Neighbours one line each, so the full record stays the thing being read.
    if before or after:
        for line in with_context(
            ["%s:%d\t(cited)" % (fname, want)], before, after, width, root
        ):
            if not line.startswith("%s:%d\t" % (fname, want)):
                print(line)
        print()
    rec = _loads(raw)
    a = rec.get("attachment") or {}
    print(
        "── %s  %s  sidechain=%s  model=%s"
        % (
            rec.get("type"),
            rec.get("timestamp"),
            rec.get("isSidechain", False),
            (rec.get("message") or {}).get("model", "-"),
        )
    )
    if rec.get("type") == "attachment":
        print("   attachment.type=%s  %s\n" % (a.get("type"), a.get("filename") or ""))
        print(body_attachment(rec))
        return 0
    if rec.get("type") == "queue-operation":
        # The text is at the top level here, not under `message`.
        print("   operation=%s\n" % (rec.get("operation") or "?"))
        print(rec.get("content") or "")
        return 0
    msg = rec.get("message") or {}
    c = msg.get("content")
    if isinstance(c, list):
        for b in c:
            t = b.get("type")
            if t == "text":
                print(b.get("text", ""))
            elif t == "thinking":
                print("[thinking] " + (b.get("thinking") or ""))
            elif t == "tool_use":
                print(f"[tool_use {b.get('name')}] " + json.dumps(b.get("input")))
            elif t == "tool_result":
                print("[tool_result] " + json.dumps(b.get("content"))[:4000])
            else:
                print(f"[{t}]")
    else:
        print(c)


def cmd_span(target, root, session, width, tslen, only, limit, subagents=False,
             since="", until=""):
    """Every record between two points, in order, whatever its type.

    The question a transcript is usually asked is not "where does this word
    appear" but "what happened between here and there". Built from line numbers
    rather than a regex because the anchors come from a previous search, and a
    second regex would silently drop the turns that make the span readable.
    """
    fname, _, rng = target.rpartition(":")
    lo_s, _, hi_s = rng.partition("-")
    if (since or until) and rng in ("", "-"):
        lo_s, hi_s = "1", "999999999"   # the whole file; the clock narrows it
    if not (lo_s.isdigit() and hi_s.isdigit()):
        print("usage: tx span [FILE:]LO-HI   (bare LO-HI needs -s)", file=sys.stderr)
        return 2
    lo, hi = int(lo_s), int(hi_s)
    if lo > hi:
        print("tx: span starts after it ends", file=sys.stderr)
        return 2

    paths = [p for p in discover(root, session, subagents)
             if not fname or os.path.basename(p) == fname]
    if not paths:
        print("tx: no transcript matching %s" % (fname or session or "(any)"), file=sys.stderr)
        return 2
    if len(paths) > 1:
        print("tx: span addresses one transcript; %d match. Narrow with -s."
              % len(paths), file=sys.stderr)
        return 2

    path = paths[0]
    base = os.path.basename(path)
    rows = []
    with open(path, "rb") as fh:
        for i, raw in enumerate(fh, 1):
            if i < lo:
                continue
            if i > hi:
                break
            try:
                rec = _loads(raw)
            except ValueError:
                continue
            if not isinstance(rec, dict):
                continue
            if since or until:
                when_ts = full_stamp(rec)
                if (since and when_ts < since) or (until and when_ts > until):
                    continue
            shown = render_context(rec, width, tslen)
            if shown is None:
                continue
            if only and shown[1].strip() not in only:
                continue
            rows.append(ROW % (field(base), i, shown[0], field(shown[1]),
                               shown[2]))
    rows = dedupe_echo(rows)
    emit(rows, limit,
         "--only user, a shorter LO-HI" if not only else "a shorter LO-HI")
    return 0 if rows else 1


SEARCH = {"say", "said", "typed", "attached", "ran", "used", "when"}

EPILOG = """\
SELECT what to search
  say       everything the operator put in: their prose AND attached files
  typed     their prose alone          attached  their files alone
  said      assistant text + thinking
  ran       Bash command text          used      any tool call (name + input)
  when      date histogram

READ what to print
  span [FILE:]LO-HI   every record between two points, in order, any type
                      `span :` with --since/--until spans by clock instead
  cite FILE:LINE      one record in full; `-` reads citations from stdin
  files               the transcripts a search would scan

NARROW
  --here          this conversation (CLAUDE_CODE_SESSION_ID)
  -s STR          one conversation: id prefix, matched on name or directory
  --subagents     also the agents that conversation dispatched (off by default)
  --from/--to N   line bounds inside it
  --since/--until WHEN   time bounds, in the format the output prints:
                  `2026-09-17` (the whole day) or `2026-09-17 04:18`. UTC
  -w              word boundaries: `mise`, not `compromise`
  --only TAGS     span only: user,asst,file,tool,sys
  -A/-B/-C N      context around each hit, counted in TURNS not raw records

OUTPUT  default: grouped columns `LINE TIME ROLE TEXT`, transcript named
        once per group. Machine formats are never capped or truncated and
        put their notes on stderr; fields are transcript,line,time,role,text.
  -n N       cap, default 40. Cut from the MIDDLE and marked there. -n 0 = all
  -T         time to the second        -W N   text width
  -oj/--json one JSON object per line. A tool call becomes {tool, input},
             nested, not a JSON string inside a string
  -on/--toon one TOON table (reference encoder): names once, a row each.
             ~28% fewer
             tokens than JSON here. Cells quoted only where needed
  --tsv      one record per line, tab separated
  (`-ot` is deliberately not a flag: it reads as either toon or tsv)

RECIPES
  tx typed mise -s 25d17901 -w              did they say it, in this session
  tx span : --here --only user --since '2026-09-17 04:15' --until 04:22 ...
                                            their turns in a time window
  tx said ERROR --tsv | cut -f1,2 | tx cite -   read every hit in full
  tx typed deploy --here -on             the cheapest form to hand to a model
  tx ran 'rm -rf' -C 2                      the exchange around each one

OTHER  -j N jobs   --root DIR corpus root

EXIT  0 found   1 nothing matched   2 error
env   TX_ROOT (default ~/.claude/projects)  TX_JOBS  TX_WIDTH  TX_LIMIT
"""


def render_context(rec, width, tslen=10):
    """(time, role, text) for ANY record, or None where it says nothing.

    Mode-agnostic on purpose: the point of context on a transcript is to see
    the EXCHANGE around a hit, so a `say` match shows the assistant's reply and
    the tool calls that followed rather than only more operator prose.
    """
    rtype = rec.get("type") or "?"
    ts = stamp(rec, tslen)
    # A queued message is one of the operator's turns; it reads as `user`.
    tag = {"user": "user", "assistant": "asst", "attachment": "file",
           "system": "sys ", "queue-operation": "user"}.get(rtype, rtype[:4])
    if rtype == "queue-operation":
        body = body_queued(rec)
        if body is None or INJECTED.search(body):
            return None
        return (ts, tag, cut(body, width))
    if rtype == "attachment":
        # Not every attachment is an operator file: the rest carry no filename
        # and a null body, which would otherwise print as the string "null".
        a = rec.get("attachment") or {}
        name = os.path.basename(a.get("filename") or "") or (a.get("type") or "attachment")
        body = body_attachment(rec) or ""
        if body in ("null", "None"):
            body = ""
        # A label with no body is a marker, not a turn.
        if not body.strip():
            return None
        return (ts, tag, "[%s] %s" % (field(name), cut(body, width)))
    if rtype == "system":
        # These carry no message; the subtype is the whole of what they say.
        label = rec.get("subtype") or rec.get("level") or "system"
        text = rec.get("content") or ""
        if not isinstance(text, str):
            text = json.dumps(text)
        if not text.strip():
            return None
        return (ts, tag, "<%s> %s" % (field(label), cut(text, width)))
    text = body_message(rec)
    # Task notifications, compaction summaries and command expansions are all
    # `user` records. Tagging them sys keeps `--only user` to what the operator
    # actually said.
    if text and rtype == "user" and INJECTED.search(text):
        tag = "sys "
    if not text:
        blocks = (rec.get("message") or {}).get("content")
        parts = []
        results = 0
        if isinstance(blocks, list):
            for b in blocks:
                if not isinstance(b, dict):
                    continue
                if b.get("type") == "tool_use":
                    parts.append("[%s] %s" % (b.get("name"), json.dumps(b.get("input") or {})))
                elif b.get("type") == "tool_result":
                    results += 1
                    parts.append("[tool_result]")
        # A record carrying only tool results is the harness answering the
        # model, not a turn either party took. Its own tag keeps it out of
        # `--only user`, where it is pure noise, without hiding it.
        if results and len(parts) == results:
            tag = "tool"
        text = " ".join(parts)
    # Bookkeeping records (session mode, latches, titles) carry no text.
    if not text:
        return None
    return (ts, tag, cut(text, width))


def with_context(hits, before, after, width, root, session=None, tslen=10,
                 subagents=False):
    """grep's -A/-B/-C, over RECORDS.

    A transcript line IS one record, so the unit matches grep's; the
    neighbours are whole turns. Context is resolved in a SECOND pass over the
    files rather than carried through the scan, so it cannot be truncated by
    the chunk boundaries the parallel scan splits large files on.

    Match lines keep their original rendering, so adding context never changes
    what a hit looks like; only the surrounding lines are the generic form.
    Groups are separated by `--`, as grep does it.
    """
    hit_text: dict[str, dict[int, str]] = {}
    want: dict[str, set[int]] = {}
    for h in hits:
        row = parse_row(h)
        if row is None:
            continue
        name, n = row[0], row[1]
        hit_text.setdefault(name, {})[n] = h
        # Context counts TURNS, not raw lines: a literal +/-N window is
        # mostly bookkeeping records that render to nothing. Read a wide band,
        # keep the N nearest records that say something.
        band = max(before, after) * SPREAD + SPREAD
        want.setdefault(name, set()).update(
            range(max(1, n - band), n + band + 1))

    paths = {os.path.basename(p): p for p in discover(root, session, subagents)}
    out = []
    for name in sorted(want):
        path = paths.get(name)
        if path is None:
            continue
        wanted, marks = want[name], hit_text.get(name, {})
        got = {}
        with open(path, "rb") as fh:
            for i, raw in enumerate(fh, 1):
                if i in wanted:
                    got[i] = raw
        # Everything in the band that says something, in order.
        speaking = []
        for i in sorted(wanted):
            if i in marks:
                speaking.append((i, marks[i], True))
                continue
            raw = got.get(i)
            if raw is None:
                continue
            try:
                rec = _loads(raw)
            except ValueError:
                continue
            shown = render_context(rec, width, tslen)
            if shown is None:
                continue
            speaking.append((i, ROW % (field(name), i, shown[0],
                                       field(shown[1]), shown[2]), False))

        # Keep the requested number of TURNS around each hit, not raw lines.
        keep = set()
        for pos, (_i, _line, is_hit) in enumerate(speaking):
            if not is_hit:
                continue
            lo_i = max(0, pos - before)
            hi_i = min(len(speaking), pos + after + 1)
            keep.update(range(lo_i, hi_i))

        prev = None
        for pos in sorted(keep):
            i, line, _ = speaking[pos]
            if prev is not None and pos != prev + 1:
                out.append("--")
            out.append(line)
            prev = pos
    return out


def main():
    ap = argparse.ArgumentParser(
        prog="tx",
        # Spelled out: the options are documented in the epilog, and a
        # generated usage line would omit them.
        usage="tx SUBCOMMAND [REGEX | FILE:LINE | [FILE:]LO-HI] [options]",
        description="Search Claude Code session transcripts.",
        epilog=EPILOG,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument(
        "sub", choices=sorted(SEARCH | {"cite", "files", "span"}), metavar="SUBCOMMAND"
    )
    ap.add_argument("arg", nargs="?",
                    help=argparse.SUPPRESS)
    ap.add_argument("--only", metavar="TAGS",
                    help=argparse.SUPPRESS)
    # -w is grep's word-regexp everywhere else; width takes -W.
    ap.add_argument("-W", "--width", type=int, default=WIDTH, help=argparse.SUPPRESS)
    ap.add_argument("-j", "--jobs", type=int, default=JOBS)
    ap.add_argument("--root", default=ROOT)
    ap.add_argument("-s", "--session", metavar="STR",
                    help=argparse.SUPPRESS)
    ap.add_argument("--here", action="store_true",
                    help=argparse.SUPPRESS)
    ap.add_argument("--subagents", action="store_true",
                    help=argparse.SUPPRESS)
    ap.add_argument("--since", metavar="WHEN", help=argparse.SUPPRESS)
    ap.add_argument("--until", metavar="WHEN", help=argparse.SUPPRESS)
    ap.add_argument("--from", dest="lo", type=int, default=0, metavar="N",
                    help=argparse.SUPPRESS)
    ap.add_argument("--to", dest="hi", type=int, default=0, metavar="N",
                    help=argparse.SUPPRESS)
    ap.add_argument("-w", "--word", action="store_true",
                    help=argparse.SUPPRESS)
    ap.add_argument("-T", "--time", action="store_true",
                    help=argparse.SUPPRESS)
    ap.add_argument("-n", "--limit", type=int, default=LIMIT, metavar="N",
                    help=argparse.SUPPRESS)
    fmt = ap.add_mutually_exclusive_group()
    fmt.add_argument("-o", "--output",
                     choices=("plain", "tsv", "json", "toon"), default="plain",
                     help=argparse.SUPPRESS)
    # No -ot: it reads as either toon or tsv. `-o t` fails with the choices.
    fmt.add_argument("--tsv", dest="fmt_tsv", action="store_true",
                     help=argparse.SUPPRESS)
    fmt.add_argument("--json", "-oj", dest="fmt_json", action="store_true",
                     help=argparse.SUPPRESS)
    fmt.add_argument("--toon", "-on", dest="fmt_toon", action="store_true",
                     help=argparse.SUPPRESS)
    ap.add_argument("-A", "--after", type=int, default=0, metavar="N",
                    help=argparse.SUPPRESS)
    ap.add_argument("-B", "--before", type=int, default=0, metavar="N",
                    help=argparse.SUPPRESS)
    ap.add_argument("-C", "--context", type=int, default=None, metavar="N",
                    help=argparse.SUPPRESS)
    args = ap.parse_args()
    if args.context is not None:
        args.after = args.before = args.context
    if (args.after or args.before) and args.after >= 0 and args.before >= 0:
        if args.sub in ("when", "files", "span"):
            ap.error("-A/-B/-C do not apply to %s" % args.sub)
    if args.after < 0 or args.before < 0:
        ap.error("-A/-B/-C take a non-negative count")

    try:
        args.since = time_bound(args.since, False) if args.since else ""
        args.until = time_bound(args.until, True) if args.until else ""
    except ValueError as e:
        ap.error("--since/--until: %s" % e)
    if args.since and args.until and args.since > args.until:
        ap.error("--since is after --until")

    if args.lo < 0 or args.hi < 0:
        ap.error("--from/--to take a non-negative line number")
    if args.lo and args.hi and args.lo > args.hi:
        ap.error("--from is past --to")

    global MACHINE
    MACHINE = ("tsv" if args.fmt_tsv else "json" if args.fmt_json
               else "toon" if args.fmt_toon else args.output)
    if MACHINE != "plain":
        # Full values for a parser; whitespace is still collapsed, which is
        # what keeps the columns framed.
        args.width = 1 << 30

    if args.here:
        if args.session:
            ap.error("--here and -s name the same thing; use one")
        args.session = here_session()
        if not args.session:
            print("tx: --here needs CLAUDE_CODE_SESSION_ID, which is unset",
                  file=sys.stderr)
            return 2

    if args.sub == "files":
        found = sorted(discover(args.root, args.session, args.subagents))
        for p in found:
            print(p)
        return 0 if found else 1
    if args.arg is None:
        ap.error(f"{args.sub} needs an argument")
    if args.sub == "cite":
        if args.arg == "-":
            targets = citations_from_stdin()
            if not targets:
                print("tx: no FILE:LINE citations on stdin", file=sys.stderr)
                return 1
            rc = 0
            for i, t in enumerate(targets):
                if i:
                    print()
                rc = cmd_cite(t, args.root, args.before, args.after,
                              args.width) or rc
            return rc
        return cmd_cite(args.arg, args.root, args.before, args.after, args.width)
    if args.sub == "span":
        only = {t.strip() for t in args.only.split(",")} if args.only else None
        return cmd_span(args.arg, args.root, args.session, args.width,
                        19 if args.time else 10, only, args.limit,
                        args.subagents, args.since, args.until)

    pattern = r"\b(?:%s)\b" % args.arg if args.word else args.arg
    try:
        re.compile(pattern)
    except re.error as e:
        print(f"tx: bad regex: {e}", file=sys.stderr)
        return 2

    # The histogram counts days whatever the stamp width, or every bucket is
    # one record.
    tslen = 10 if args.sub == "when" else (19 if args.time else 10)
    hits = run(args.sub, pattern, args.width, args.jobs, args.root,
               args.session, args.lo, args.hi, tslen, args.subagents,
               args.since, args.until)
    if args.sub == "when":
        # No dedup here: deduping before counting collapses every date to 1.
        for date, n in sorted(Counter(hits).items()):
            print(f"{n:7d} {date}")
        return 0 if hits else 1

    rows = dedupe_echo(sorted(set(hits), key=loc_key))
    refine = "--here or -s <id>, --from/--to" + ("" if args.word else ", -w")
    if args.after or args.before:
        rows = dedupe_echo(with_context(
            rows, args.before, args.after, args.width, args.root,
            args.session, tslen, args.subagents))
        refine = "-C 0, " + refine
    emit(rows, args.limit, refine)
    return 0 if rows else 1


def entrypoint() -> None:
    """Console-script wrapper: exit status, SIGPIPE, Ctrl-C.

    Exits 0 on hits, 1 on no match, 2 on error, 130 on interrupt.
    """
    # A closed pipe (`tx ... | head`) is the reader's decision, not an error.
    signal.signal(signal.SIGPIPE, signal.SIG_DFL)
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        sys.exit(130)


if __name__ == "__main__":
    entrypoint()
