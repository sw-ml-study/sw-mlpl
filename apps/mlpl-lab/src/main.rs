// mlpl-lab is a placeholder for the future interactive experiment
// workbench described in apps/mlpl-lab/README.md. The slot is
// reserved in the workspace so the binary name is taken; the real
// implementation lands in a future saga. Until then, the binary
// supports only `-V` / `--version`.

fn main() {
    let args: Vec<String> = std::env::args().collect();
    if args.iter().any(|a| a == "-V" || a == "--version") {
        println!(
            "{} {} ({} {} {})",
            env!("CARGO_PKG_NAME"),
            env!("CARGO_PKG_VERSION"),
            env!("BUILD_HOST"),
            env!("GIT_HASH"),
            env!("BUILD_TIMESTAMP"),
        );
        return;
    }
    println!(
        "mlpl-lab v{} (placeholder; future interactive experiment workbench)",
        env!("CARGO_PKG_VERSION")
    );
    println!(
        "Build: host={} commit={} built={}",
        env!("BUILD_HOST"),
        env!("GIT_HASH"),
        env!("BUILD_TIMESTAMP"),
    );
}
