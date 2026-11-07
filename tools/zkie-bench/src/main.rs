fn main() {
    let args = std::env::args().skip(1).collect();
    std::process::exit(zkie_bench::run_cli(args));
}
