fn main() {
    let args = std::env::args().skip(1).collect();
    std::process::exit(zkie::run_cli(args));
}
