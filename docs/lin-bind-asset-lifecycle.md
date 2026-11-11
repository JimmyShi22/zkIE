# Bounded asset lifetime on the hidden prove path

## Scope

This note lists every allocation on the hidden output-head prove path that can
hold table or weight data, with its lifetime and byte upper bound. It supports
the claim that the 268,435,456 B (256 MiB) weight asset is read through the
bounded reader and is never copied into a whole-matrix allocation.

Snapshot: code basis `3b3e8ecc`, bounded-read rollback `8d32d3de`; fixture
`gpt2-seq16`, table shape `2^10` rows by `2^16` columns of `i32`, table code
`log_n = 17`, `ell = 65539`, `r = 8192`.

The inventory and every bound below apply only to this fixture path and these
listed shapes and code parameters. They are not extrapolated to other models,
sequence lengths, batch shapes or production entry points.

## Asset and reader

- The asset is opened by `BoundedAssetReader` over the read-only file. The
  mapped window is 1 MiB (`WINDOW_BYTES`); the total read budget is 1 GiB.
- `LmHeadAssetSource.buffer` is `factor * 4` bytes, where `factor = 2^k_lo`.
  With `k_lo = 6` this is **256 B**. It lives for the source lifetime and is
  reused for every row.
- `TableRowSource::row` reads each matrix row's `factor` words into that
  buffer and writes them into the caller's message slice. After the rollback
  there is no per-row `Vec` and no retained row cache.

The whole asset is 67,108,864 `i32` entries. No allocation in this path has
that length.

## Table commit

`commit_table_streaming` reads one row at a time and keeps one panel:

| Allocation | Size |
|---|---:|
| message `ell * 8` | 524,312 B |
| tail `r * 8` | 65,536 B |
| symbols `rows * panel * 8` (panel 4096) | 33,554,432 B (32 MiB) |
| codeword `n * 8` | 1,048,576 B (1 MiB) |
| column leaves `n * 32` | 4,194,304 B (4 MiB) |
| Merkle tree `2 * n * 32` | 8,388,608 B (8 MiB) |
| record answers `rows * answers * 8` | 8,192 B |

The commit's own peak check is the sum of these buffers (about 47 MiB) and is
below the commit budget. The largest single allocation is the 32 MiB panel;
the 256 MiB asset is never one buffer.

## Table retention and column opening

`StreamingTable` retains the column leaves, the tree, the salts and the name:
about 4 MiB + 8 MiB + 4 MiB + name, not the codewords. `open_columns` reads the
rows again and holds:

| Allocation | Size |
|---|---:|
| message | 524,312 B |
| tail | 65,536 B |
| symbols `positions * rows * 8` (603 positions) | 4,939,776 B |
| codeword | 1,048,576 B |
| opening values and paths | about 5 MiB |

These are released after the opening. The table row source's 256 B buffer is
the only asset buffer retained across passes.

## Reduced rows

`derive_w_rows` holds one message and one tail at a time and writes the
reduced rows:

| Allocation | Size |
|---|---:|
| reduced rows `positions * columns * 24` (16 x 65536, `Fp3`) | 25,165,824 B (24 MiB) |
| message | 524,312 B |
| tail | 65,536 B |
| activation weights `positions * matrix_rows * 24` | 393,216 B |

The reduced rows are the W messages, not a copy of the table. Each W message
is one row of the small W code; the retained W messages are far below the
table size.

## Simultaneous asset-related allocations

The asset-related allocations do not all live at once. The overlap peaks are:

| Phase | Live allocations | Total |
|---|---|---:|
| table commit | buffer 256 B + message 524,312 B + tail 65,536 B + panel 33,554,432 B + codeword 1,048,576 B + leaves 4,194,304 B + tree 8,388,608 B + salts 4,194,304 B + record 8,192 B | 51,978,520 B (~49.6 MiB) |
| column opening | buffer + message + tail + symbols 4,939,776 B + codeword 1,048,576 B + retained table 16,777,216 B + opening values about 5,242,880 B | about 28.6 MiB |
| reduced rows | buffer + message + tail + reduced rows 25,165,824 B + weights 393,216 B | about 25.0 MiB |

The largest simultaneous asset-related total is therefore below **64 MiB**,
while the asset itself is 256 MiB. The source buffer (256 B) lives for the
whole prove; the message/tail are reused; the panel, codeword, leaves/tree and
opening buffers have the lifetimes listed above. This bound covers the
asset-related copies, not the process baseline or the later closure/proximity
allocations.

## Trees and later messages


The W runtime trees, fold state, closure bits/map/slack messages and BASE rows
are derived from the reduced rows and the output relation. They have their own
plan-sized bounds and are not allocations of the weight matrix. Their sizes
are visible in the profile scopes (`runtime_tree.commit`, `first.commit_w_rows`,
`second.closure_*`, `prox.*`); none is the 268,435,456 B table.

## Measured cross-check

In the formal profile the table open/commit scopes start at about 4.2 GiB
resident and the table commit ends around 4.9 GiB, including the baseline
process, the mmap page cache and allocator slack. The prove peak later reaches
about 162 GiB, driven by the closure and proximity stages, not by the table.
The static list above rules out a whole-matrix anonymous allocation; the
profile is a cross-check, not the proof.

## Boundary

This inventory supports the listed GPT-2 `gpt2-seq16` fixture path only; it is
not extrapolated to other models, sequence lengths or production entries.

`[FACT]` No code path in the hidden prove entry allocates a `Vec` of
`element_count` entries or a buffer of 268,435,456 B for the asset.
`[ASSUMPTION]` The OS may keep mmap pages of the read-only asset in the page
cache; that is bounded by the file and the reader window, not by an anonymous
allocation. A deployment that copies the file elsewhere before mmap is outside
this path.
