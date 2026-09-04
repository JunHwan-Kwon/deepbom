use serde::Deserialize;
use sha2::{Digest, Sha256};
use std::env;
use std::ffi::OsString;
use std::fs::{self, File, OpenOptions};
use std::io::{self, Read, Write};
use std::path::{Path, PathBuf};
use std::process::{Command, ExitCode};
use std::thread;
use std::time::{Duration, Instant, SystemTime};

const VERSION: &str = env!("CARGO_PKG_VERSION");
const MATRIX_SCHEMA: &str = "deepbom.engine_matrix.v1";
const RELEASE_REPOSITORY: &str = "https://github.com/JunHwan-Kwon/deepbom";
const MAX_MATRIX_BYTES: u64 = 1024 * 1024;
const MAX_ENGINE_BYTES: u64 = 512 * 1024 * 1024;
const MAX_WASM_BYTES: u64 = 64 * 1024 * 1024;
const MAX_SELF_TEST_BYTES: u64 = 16 * 1024 * 1024;
const DOWNLOAD_CHUNK_BYTES: usize = 64 * 1024;
const RUNTIME_WASM_FILENAME: &str = "tflite_wasm_audit_bg.wasm";
const RUNTIME_SELF_TEST_FILENAME: &str = "deepbom-self-test.onnx";

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct EngineMatrix {
    schema: String,
    version: String,
    source: SourceIdentity,
    wasm: Artifact,
    self_test: Artifact,
    targets: Vec<Target>,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct SourceIdentity {
    git_commit: String,
    git_state: String,
    tag: String,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct Target {
    id: String,
    platform: String,
    arch: String,
    executable: Artifact,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct Artifact {
    filename: String,
    byte_length: u64,
    sha256: String,
}

struct InstallLock {
    path: PathBuf,
    _file: File,
}

impl Drop for InstallLock {
    fn drop(&mut self) {
        let _ = fs::remove_file(&self.path);
    }
}

fn main() -> ExitCode {
    match run() {
        Ok(code) => ExitCode::from(code.clamp(0, 255) as u8),
        Err(error) => {
            eprintln!("deepbom: {error}");
            ExitCode::from(2)
        }
    }
}

fn run() -> Result<i32, String> {
    let arguments: Vec<OsString> = env::args_os().skip(1).collect();
    if arguments.len() == 1 && matches!(arguments[0].to_str(), Some("--version") | Some("-V")) {
        println!("deepbom {VERSION}");
        return Ok(0);
    }
    if arguments.first().and_then(|value| value.to_str()) == Some("engine") {
        return engine_command(&arguments[1..]);
    }

    let installation = resolve_engine()?;
    let mut command = Command::new(&installation.executable);
    command.args(arguments);
    if let Some(asset_root) = installation.asset_root {
        command.env("DEEPBOM_RUNTIME_ASSET_DIR", asset_root);
    }
    let status = command
        .status()
        .map_err(|error| format!("could not execute verified engine: {error}"))?;
    Ok(status.code().unwrap_or(1))
}

struct EngineInstallation {
    executable: PathBuf,
    asset_root: Option<PathBuf>,
}

fn resolve_engine() -> Result<EngineInstallation, String> {
    if let Some(value) = env::var_os("DEEPBOM_ENGINE") {
        return verified_override(PathBuf::from(value));
    }
    install_or_verify_cached_engine(false)
}

fn verified_override(path: PathBuf) -> Result<EngineInstallation, String> {
    let expected = env::var("DEEPBOM_ENGINE_SHA256")
        .unwrap_or_default()
        .to_lowercase();
    validate_sha256(&expected, "DEEPBOM_ENGINE_SHA256")?;
    if !path.is_file() {
        return Err(format!("DEEPBOM_ENGINE is not a file: {}", path.display()));
    }
    verify_file(&path, None, &expected, "DEEPBOM_ENGINE")?;
    let asset_root = env::var_os("DEEPBOM_RUNTIME_ASSET_DIR").map(PathBuf::from);
    Ok(EngineInstallation {
        executable: path,
        asset_root,
    })
}

fn engine_command(arguments: &[OsString]) -> Result<i32, String> {
    let action = arguments
        .first()
        .and_then(|value| value.to_str())
        .unwrap_or("path");
    if arguments.len() > 1 {
        return Err("usage: deepbom engine [install|path|verify]".to_string());
    }
    match action {
        "install" => {
            let installation = install_or_verify_cached_engine(false)?;
            println!("{}", installation.executable.display());
        }
        "path" => {
            let installation = install_or_verify_cached_engine(false)?;
            println!("{}", installation.executable.display());
        }
        "verify" => {
            let installation = install_or_verify_cached_engine(true)?;
            println!("verified {}", installation.executable.display());
        }
        _ => return Err("usage: deepbom engine [install|path|verify]".to_string()),
    }
    Ok(0)
}

fn install_or_verify_cached_engine(
    force_remote_manifest: bool,
) -> Result<EngineInstallation, String> {
    let target_id = target_id()?;
    let root = cache_root()?.join("engines").join(VERSION).join(&target_id);
    let executable = root.join(executable_filename(&target_id));
    let asset_root = root.join("pkg");
    // The immutable release asset is versioned, while the engine's runtime
    // contract resolves the canonical wasm-pack filename inside this directory.
    let wasm = asset_root.join(runtime_wasm_filename());
    let self_test = root.join(runtime_self_test_filename());
    let matrix_path = root.join("engine-matrix.v1.json");

    if !force_remote_manifest
        && verify_cached(&matrix_path, &executable, &wasm, &self_test, &target_id).is_ok()
    {
        return Ok(EngineInstallation {
            executable,
            asset_root: Some(asset_root),
        });
    }

    fs::create_dir_all(&root).map_err(io_error("create engine cache"))?;
    let _lock = acquire_lock(&root.join(".install.lock"))?;
    if !force_remote_manifest
        && verify_cached(&matrix_path, &executable, &wasm, &self_test, &target_id).is_ok()
    {
        return Ok(EngineInstallation {
            executable,
            asset_root: Some(asset_root),
        });
    }

    eprintln!("deepbom: installing verified engine {VERSION} for {target_id}");
    let matrix_bytes = download_small(&asset_url("engine-matrix.v1.json"), MAX_MATRIX_BYTES)?;
    let matrix = parse_matrix(&matrix_bytes, &target_id)?;
    let target = matrix
        .targets
        .iter()
        .find(|item| item.id == target_id)
        .ok_or_else(|| format!("release has no engine for {target_id}"))?;
    fs::create_dir_all(&asset_root).map_err(io_error("create runtime asset cache"))?;
    download_verified(
        &asset_url(&target.executable.filename),
        &executable,
        &target.executable,
        MAX_ENGINE_BYTES,
    )?;
    download_verified(
        &asset_url(&matrix.wasm.filename),
        &wasm,
        &matrix.wasm,
        MAX_WASM_BYTES,
    )?;
    download_verified(
        &asset_url(&matrix.self_test.filename),
        &self_test,
        &matrix.self_test,
        MAX_SELF_TEST_BYTES,
    )?;
    write_atomic(&matrix_path, &matrix_bytes)?;
    set_executable(&executable)?;
    verify_cached(&matrix_path, &executable, &wasm, &self_test, &target_id)?;
    Ok(EngineInstallation {
        executable,
        asset_root: Some(asset_root),
    })
}

fn parse_matrix(bytes: &[u8], target_id: &str) -> Result<EngineMatrix, String> {
    let matrix: EngineMatrix = serde_json::from_slice(bytes)
        .map_err(|error| format!("engine matrix is invalid JSON: {error}"))?;
    if matrix.schema != MATRIX_SCHEMA || matrix.version != VERSION {
        return Err("engine matrix schema or version does not match this launcher".to_string());
    }
    let expected_tag = format!("channels-v{VERSION}");
    if matrix.source.tag != expected_tag
        || matrix.source.git_state != "clean"
        || matrix.source.git_commit.len() != 40
        || !matrix
            .source
            .git_commit
            .bytes()
            .all(|byte| byte.is_ascii_hexdigit())
    {
        return Err("engine matrix source identity is invalid".to_string());
    }
    if matrix.targets.len() != 6 {
        return Err("engine matrix must contain exactly six release targets".to_string());
    }
    let mut ids = matrix
        .targets
        .iter()
        .map(|item| item.id.as_str())
        .collect::<Vec<_>>();
    ids.sort_unstable();
    ids.dedup();
    if ids.len() != matrix.targets.len() {
        return Err("engine matrix contains duplicate target ids".to_string());
    }
    validate_artifact(&matrix.wasm, &wasm_filename(), MAX_WASM_BYTES)?;
    validate_artifact(
        &matrix.self_test,
        &self_test_filename(),
        MAX_SELF_TEST_BYTES,
    )?;
    for target in &matrix.targets {
        let expected = executable_filename(&target.id);
        validate_artifact(&target.executable, &expected, MAX_ENGINE_BYTES)?;
        validate_target_identity(target)?;
    }
    if !matrix.targets.iter().any(|item| item.id == target_id) {
        return Err(format!("engine matrix does not support {target_id}"));
    }
    Ok(matrix)
}

fn validate_target_identity(target: &Target) -> Result<(), String> {
    let expected = match target.id.as_str() {
        "windows-x64" => ("windows", "x64"),
        "windows-arm64" => ("windows", "arm64"),
        "linux-x64" => ("linux", "x64"),
        "linux-arm64" => ("linux", "arm64"),
        "macos-x64" => ("macos", "x64"),
        "macos-arm64" => ("macos", "arm64"),
        _ => return Err(format!("unsupported engine target {}", target.id)),
    };
    if target.platform != expected.0 || target.arch != expected.1 {
        return Err(format!(
            "engine target {} has inconsistent platform identity",
            target.id
        ));
    }
    Ok(())
}

fn validate_artifact(artifact: &Artifact, expected_name: &str, maximum: u64) -> Result<(), String> {
    if artifact.filename != expected_name
        || artifact.byte_length == 0
        || artifact.byte_length > maximum
    {
        return Err(format!(
            "invalid release artifact record for {expected_name}"
        ));
    }
    validate_sha256(&artifact.sha256, expected_name)
}

fn verify_cached(
    matrix_path: &Path,
    executable: &Path,
    wasm: &Path,
    self_test: &Path,
    target_id: &str,
) -> Result<(), String> {
    let bytes = fs::read(matrix_path).map_err(io_error("read cached engine matrix"))?;
    if bytes.len() as u64 > MAX_MATRIX_BYTES {
        return Err("cached engine matrix exceeds its size bound".to_string());
    }
    let matrix = parse_matrix(&bytes, target_id)?;
    let target = matrix
        .targets
        .iter()
        .find(|item| item.id == target_id)
        .ok_or_else(|| format!("cached engine matrix has no {target_id} target"))?;
    verify_file(
        executable,
        Some(target.executable.byte_length),
        &target.executable.sha256,
        "engine",
    )?;
    verify_file(
        wasm,
        Some(matrix.wasm.byte_length),
        &matrix.wasm.sha256,
        "TFLite WASM",
    )?;
    verify_file(
        self_test,
        Some(matrix.self_test.byte_length),
        &matrix.self_test.sha256,
        "self-test probe",
    )
}

fn download_small(url: &str, maximum: u64) -> Result<Vec<u8>, String> {
    let mut response = ureq::get(url)
        .header("User-Agent", &format!("deepbom/{VERSION}"))
        .call()
        .map_err(|error| format!("download failed for {url}: {error}"))?;
    let mut bytes = Vec::new();
    response
        .body_mut()
        .as_reader()
        .take(maximum + 1)
        .read_to_end(&mut bytes)
        .map_err(|error| format!("could not read {url}: {error}"))?;
    if bytes.len() as u64 > maximum {
        return Err(format!("download exceeded {maximum} bytes: {url}"));
    }
    Ok(bytes)
}

fn download_verified(
    url: &str,
    destination: &Path,
    artifact: &Artifact,
    maximum: u64,
) -> Result<(), String> {
    if artifact.byte_length > maximum {
        return Err(format!("{} exceeds its download bound", artifact.filename));
    }
    let temporary = temporary_path(destination);
    let result = (|| {
        let mut response = ureq::get(url)
            .header("User-Agent", &format!("deepbom/{VERSION}"))
            .call()
            .map_err(|error| format!("download failed for {url}: {error}"))?;
        let mut reader = response
            .body_mut()
            .as_reader()
            .take(artifact.byte_length + 1);
        let mut output = File::create(&temporary).map_err(io_error("create temporary download"))?;
        let mut digest = Sha256::new();
        let mut total = 0_u64;
        // Keep the download buffer on the heap. A 1 MiB stack allocation exhausts
        // the default Windows main-thread stack before the first network read.
        let mut buffer = vec![0_u8; DOWNLOAD_CHUNK_BYTES];
        loop {
            let count = reader
                .read(&mut buffer)
                .map_err(io_error("read release artifact"))?;
            if count == 0 {
                break;
            }
            total += count as u64;
            if total > artifact.byte_length {
                return Err(format!("{} exceeded its declared size", artifact.filename));
            }
            digest.update(&buffer[..count]);
            output
                .write_all(&buffer[..count])
                .map_err(io_error("write release artifact"))?;
        }
        output
            .sync_all()
            .map_err(io_error("sync release artifact"))?;
        let observed = format!("{:x}", digest.finalize());
        if total != artifact.byte_length || observed != artifact.sha256 {
            return Err(format!(
                "{} failed its size or SHA-256 check",
                artifact.filename
            ));
        }
        replace_file(&temporary, destination)
    })();
    if result.is_err() {
        let _ = fs::remove_file(&temporary);
    }
    result
}

fn write_atomic(destination: &Path, bytes: &[u8]) -> Result<(), String> {
    let temporary = temporary_path(destination);
    let result = (|| {
        let mut output = File::create(&temporary).map_err(io_error("create temporary manifest"))?;
        output
            .write_all(bytes)
            .map_err(io_error("write engine matrix"))?;
        output.sync_all().map_err(io_error("sync engine matrix"))?;
        replace_file(&temporary, destination)
    })();
    if result.is_err() {
        let _ = fs::remove_file(&temporary);
    }
    result
}

fn replace_file(source: &Path, destination: &Path) -> Result<(), String> {
    if destination.exists() {
        fs::remove_file(destination).map_err(io_error("replace cached artifact"))?;
    }
    fs::rename(source, destination).map_err(io_error("commit cached artifact"))
}

fn verify_file(
    path: &Path,
    expected_size: Option<u64>,
    expected_sha: &str,
    label: &str,
) -> Result<(), String> {
    let metadata = fs::metadata(path).map_err(io_error("read cached artifact metadata"))?;
    if !metadata.is_file() || expected_size.is_some_and(|size| size != metadata.len()) {
        return Err(format!("cached {label} has the wrong type or size"));
    }
    let mut input = File::open(path).map_err(io_error("open cached artifact"))?;
    let mut digest = Sha256::new();
    io::copy(&mut input, &mut digest).map_err(io_error("hash cached artifact"))?;
    if format!("{:x}", digest.finalize()) != expected_sha {
        return Err(format!("cached {label} failed its SHA-256 check"));
    }
    Ok(())
}

fn acquire_lock(path: &Path) -> Result<InstallLock, String> {
    let started = Instant::now();
    loop {
        match OpenOptions::new().write(true).create_new(true).open(path) {
            Ok(mut file) => {
                writeln!(file, "{}", std::process::id()).map_err(io_error("write install lock"))?;
                return Ok(InstallLock {
                    path: path.to_path_buf(),
                    _file: file,
                });
            }
            Err(error) if error.kind() == io::ErrorKind::AlreadyExists => {
                let stale = fs::metadata(path)
                    .and_then(|value| value.modified())
                    .ok()
                    .and_then(|time| SystemTime::now().duration_since(time).ok())
                    .is_some_and(|age| age > Duration::from_secs(300));
                if stale {
                    let _ = fs::remove_file(path);
                    continue;
                }
                if started.elapsed() > Duration::from_secs(30) {
                    return Err("timed out waiting for another engine installation".to_string());
                }
                thread::sleep(Duration::from_millis(200));
            }
            Err(error) => return Err(format!("could not create engine install lock: {error}")),
        }
    }
}

fn cache_root() -> Result<PathBuf, String> {
    if let Some(value) = env::var_os("DEEPBOM_HOME") {
        return Ok(PathBuf::from(value));
    }
    let home = env::var_os(if cfg!(windows) { "USERPROFILE" } else { "HOME" })
        .ok_or_else(|| "DEEPBOM_HOME and the user home directory are unavailable".to_string())?;
    Ok(PathBuf::from(home).join(".deepbom"))
}

fn target_id() -> Result<String, String> {
    let platform = match env::consts::OS {
        "windows" => "windows",
        "linux" => "linux",
        "macos" => "macos",
        other => return Err(format!("unsupported operating system {other}")),
    };
    let arch = match env::consts::ARCH {
        "x86_64" => "x64",
        "aarch64" => "arm64",
        other => return Err(format!("unsupported architecture {other}")),
    };
    Ok(format!("{platform}-{arch}"))
}

fn executable_filename(target_id: &str) -> String {
    format!(
        "deepbom-core-{target_id}{}",
        if target_id.starts_with("windows-") {
            ".exe"
        } else {
            ""
        }
    )
}

fn wasm_filename() -> String {
    format!("tflite_wasm_audit_bg-{VERSION}.wasm")
}

fn self_test_filename() -> String {
    format!("deepbom-self-test-{VERSION}.onnx")
}

fn runtime_wasm_filename() -> &'static str {
    RUNTIME_WASM_FILENAME
}

fn runtime_self_test_filename() -> &'static str {
    RUNTIME_SELF_TEST_FILENAME
}

fn asset_url(filename: &str) -> String {
    format!("{RELEASE_REPOSITORY}/releases/download/channels-v{VERSION}/{filename}")
}

fn validate_sha256(value: &str, label: &str) -> Result<(), String> {
    if value.len() != 64
        || !value
            .bytes()
            .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
    {
        return Err(format!("{label} is not a lowercase 64-character SHA-256"));
    }
    Ok(())
}

fn temporary_path(destination: &Path) -> PathBuf {
    destination.with_extension(format!(
        "{}.{}.tmp",
        destination
            .extension()
            .and_then(|value| value.to_str())
            .unwrap_or("download"),
        std::process::id()
    ))
}

#[cfg(unix)]
fn set_executable(path: &Path) -> Result<(), String> {
    use std::os::unix::fs::PermissionsExt;
    let mut permissions = fs::metadata(path)
        .map_err(io_error("read engine permissions"))?
        .permissions();
    permissions.set_mode(0o755);
    fs::set_permissions(path, permissions).map_err(io_error("set engine executable permission"))
}

#[cfg(not(unix))]
fn set_executable(_path: &Path) -> Result<(), String> {
    Ok(())
}

fn io_error(context: &'static str) -> impl FnOnce(io::Error) -> String {
    move |error| format!("{context}: {error}")
}

#[cfg(test)]
mod tests {
    use super::*;

    fn artifact(filename: String) -> Artifact {
        Artifact {
            filename,
            byte_length: 1,
            sha256: "a".repeat(64),
        }
    }

    fn matrix_json() -> Vec<u8> {
        let targets = [
            ("windows-x64", "windows", "x64"),
            ("windows-arm64", "windows", "arm64"),
            ("linux-x64", "linux", "x64"),
            ("linux-arm64", "linux", "arm64"),
            ("macos-x64", "macos", "x64"),
            ("macos-arm64", "macos", "arm64"),
        ]
        .into_iter()
        .map(|(id, platform, arch)| {
            serde_json::json!({
                "id": id,
                "platform": platform,
                "arch": arch,
                "executable": {
                    "filename": executable_filename(id),
                    "byte_length": 1,
                    "sha256": "a".repeat(64)
                }
            })
        })
        .collect::<Vec<_>>();
        serde_json::to_vec(&serde_json::json!({
            "schema": MATRIX_SCHEMA,
            "version": VERSION,
            "source": {
                "git_commit": "b".repeat(40),
                "git_state": "clean",
                "tag": format!("channels-v{VERSION}")
            },
            "wasm": {
                "filename": wasm_filename(),
                "byte_length": 1,
                "sha256": "a".repeat(64)
            },
            "self_test": {
                "filename": self_test_filename(),
                "byte_length": 1,
                "sha256": "a".repeat(64)
            },
            "targets": targets
        }))
        .unwrap()
    }

    #[test]
    fn release_asset_names_are_platform_bound() {
        assert_eq!(
            executable_filename("windows-x64"),
            "deepbom-core-windows-x64.exe"
        );
        assert_eq!(
            executable_filename("linux-arm64"),
            "deepbom-core-linux-arm64"
        );
        assert_eq!(
            wasm_filename(),
            format!("tflite_wasm_audit_bg-{VERSION}.wasm")
        );
        assert_eq!(runtime_wasm_filename(), "tflite_wasm_audit_bg.wasm");
        assert_eq!(
            self_test_filename(),
            format!("deepbom-self-test-{VERSION}.onnx")
        );
        assert_eq!(runtime_self_test_filename(), "deepbom-self-test.onnx");
        assert_ne!(wasm_filename(), runtime_wasm_filename());
        assert_ne!(self_test_filename(), runtime_self_test_filename());
    }

    #[test]
    fn sha256_contract_rejects_non_hex_and_wrong_length() {
        assert!(validate_sha256(&"a".repeat(64), "test").is_ok());
        assert!(validate_sha256(&"g".repeat(64), "test").is_err());
        assert!(validate_sha256(&"A".repeat(64), "test").is_err());
        assert!(validate_sha256(&"a".repeat(63), "test").is_err());
    }

    #[test]
    fn matrix_contract_accepts_only_complete_bound_targets() {
        assert!(parse_matrix(&matrix_json(), "windows-x64").is_ok());
        let mut value: serde_json::Value = serde_json::from_slice(&matrix_json()).unwrap();
        value["targets"][0]["platform"] = serde_json::json!("linux");
        assert!(parse_matrix(&serde_json::to_vec(&value).unwrap(), "windows-x64").is_err());
    }

    #[test]
    fn artifact_contract_is_size_name_and_digest_bound() {
        let expected = wasm_filename();
        assert!(validate_artifact(&artifact(expected.clone()), &expected, 2).is_ok());
        assert!(validate_artifact(&artifact("other.wasm".to_string()), &expected, 2).is_err());
        assert!(validate_artifact(&artifact(expected.clone()), &expected, 0).is_err());
    }

    #[test]
    fn download_buffer_stays_below_windows_stack_pressure() {
        assert!(DOWNLOAD_CHUNK_BYTES <= 64 * 1024);
    }
}
