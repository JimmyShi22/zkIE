# zkIE — Project Notes

## Hard constraint: where to run builds

Local security software blocks (SIGKILLs) build-script execution (`build.rs`, e.g.
`libc`, `crossbeam-utils`, `num-traits`, etc.) when the process runs under
`/tmp` / `/private/tmp` (including this session's scratchpad directory). This is a
hard limit — do not run `cargo build`/`cargo test`/anything that compiles a crate
with a build script from a `/tmp`-based path.

**Always run cargo commands from inside this project directory**
(`/Users/jimmyshi/code/zkie/...`), not from `/tmp` or the scratchpad. If a
throwaway spike is needed, create it as a subdirectory inside the project (e.g.
`.spike-test/`, gitignored) rather than under `/tmp`.

Alternative if the above ever fails again: run the build inside a Docker
container — container execution is not subject to this restriction.
