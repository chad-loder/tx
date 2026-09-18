# tx

Ad-hoc search over Claude Code session transcripts.

Claude Code writes every conversation to `~/.claude/projects/**/*.jsonl`. `tx` searches
that corpus and prints stable `<transcript>:<line>` citations you can feed back into
`tx cite` to read any record in full.

Seven properties of the corpus make a naive `grep` silently wrong — subagent transcripts
nested a directory deeper, project directories whose names begin with `-` (so shell globs
and `find` parse them as flags), `user` records that are not the operator, operator text
that arrives as a separate `attachment` or `queue-operation` record, and message content
that is sometimes a list of blocks and sometimes a bare string. Each one produces a
confident empty result. `tx` handles all of them; the module docstring in
[src/tx/cli.py](src/tx/cli.py) spells out each trap.

## Install

```bash
uv tool install --python 3.14t git+https://github.com/chad-loder/tx
```

From a local checkout:

```bash
uv tool install --python 3.14t ~/dev/tx
```

Either form puts the `tx` executable in uv's tool bin directory — `$XDG_BIN_HOME`, else
`$XDG_DATA_HOME/../bin`, else `~/.local/bin` — which is the same location on macOS and
Linux. Run `uv tool update-shell` once if that directory is not already on `PATH`.

### Developer mode

An editable install points the same `tx` command at your working tree, so edits to
`src/tx/cli.py` take effect on the next run with no reinstall:

```bash
uv tool install --python 3.14t --editable ~/dev/tx
```

`uv tool install --force` switches between the two at any time.

### Python builds

`tx` scans the corpus across a thread pool sized to the machine's performance cores, so a
free-threaded interpreter (`3.14t`) is the fast path and what `.python-version` selects.
It runs correctly on any CPython >= 3.13; the GIL just serializes the scan.

orjson publishes no free-threaded wheel, and its source build refuses a free-threaded
target unless `ORJSON_BUILD_FREETHREADED=1` is set. uv exports it from `[tool.uv.env]`
during the build; a manual `pip install` into a free-threaded environment needs it in the
environment, plus rustc >= 1.95.

## Use

```bash
tx say 'headroom'                    # everything the operator put in, prose and attachments
tx typed deploy --here               # operator prose alone, this conversation only
tx said 'prefix cache'               # assistant text and thinking
tx ran 'rm -rf' -C 2                 # Bash commands, with two turns of context each side
tx used Edit --since 2026-09-17      # tool calls by name and input
tx when 'timeout'                    # date histogram
tx cite ~/.claude/projects/x.jsonl:412
tx span 100-140 --only user,asst     # every record across a line span
```

Selectors: `say` `typed` `attached` `said` `ran` `used` `when` `span` `cite` `files`.
Narrowing: `--here`, `-s <session>`, `--subagents`, `--from/--to`, `--since/--until`,
`-w`, `--only`, `-A/-B/-C`. Output: grouped columns by default, `-oj` for JSONL, `-on`
for a TOON table (~28% fewer tokens than JSON), `--tsv` for cut/awk. `-n 0` lifts the
40-record cap.

Exit status is 0 on hits, 1 on no match, 2 on error — so `tx typed foo >/dev/null &&
echo seen` works in a script.

## Environment

| Variable | Default | Meaning |
|---|---|---|
| `TX_ROOT` | `~/.claude/projects` | Corpus root to scan |
| `TX_WIDTH` | `130` | Text column width |
| `TX_LIMIT` | `40` | Default record cap |

`--root`, `-W` and `-n` override these per invocation. `-j` sets the thread count, which
defaults to the machine's performance-core count.

<!-- pypi-end -->

## Develop

```bash
uv sync                  # resolve the dev environment
uv run pytest            # tests
uv run ruff check .      # lint
uv run tx --help         # run from the tree without installing
```

## License

MIT — see [LICENSE](LICENSE).
