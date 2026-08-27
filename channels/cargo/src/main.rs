use std::env;
use std::path::{Path, PathBuf};
use std::process::{Command, ExitCode};

fn engine_path() -> Result<PathBuf, String> {
    if let Some(value) = env::var_os("DEEPBOM_ENGINE") {
        let path = PathBuf::from(value);
        if path.is_file() {
            return Ok(path);
        }
        return Err(format!("DEEPBOM_ENGINE is not a file: {}", path.display()));
    }

    let executable = env::current_exe().map_err(|error| error.to_string())?;
    let filename = if cfg!(windows) { "deepbom-core.exe" } else { "deepbom-core" };
    let adjacent = executable.parent().unwrap_or_else(|| Path::new(".")).join(filename);
    if adjacent.is_file() {
        return Ok(adjacent);
    }

    Err("canonical DEEPBOM engine not found; install the platform engine beside this launcher or set DEEPBOM_ENGINE to a verified engine path".to_string())
}

fn main() -> ExitCode {
    let engine = match engine_path() {
        Ok(path) => path,
        Err(error) => {
            eprintln!("deepbom: {error}");
            return ExitCode::from(2);
        }
    };
    match Command::new(engine).args(env::args_os().skip(1)).status() {
        Ok(status) => ExitCode::from(status.code().unwrap_or(1).clamp(0, 255) as u8),
        Err(error) => {
            eprintln!("deepbom: could not execute canonical engine: {error}");
            ExitCode::from(2)
        }
    }
}
