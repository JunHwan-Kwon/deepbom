use std::env;
use std::fs;
use std::path::PathBuf;
use std::time::{SystemTime, UNIX_EPOCH};

const DEFAULT_EXPIRY_DAYS: u64 = 30;
const MILLIS_PER_SECOND: u64 = 1_000;
const SECONDS_PER_DAY: u64 = 86_400;

fn main() {
    println!("cargo:rerun-if-env-changed=APP_EXPIRES_AT");
    println!("cargo:rerun-if-env-changed=APP_EXPIRES_AT_EPOCH_MS");
    println!("cargo:rerun-if-env-changed=APP_NOT_BEFORE_EPOCH_MS");
    println!("cargo:rerun-if-env-changed=APP_EXPIRY_DAYS");

    let now_secs = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .expect("system clock must be after unix epoch")
        .as_secs();

    let expiry_days = env::var("APP_EXPIRY_DAYS")
        .ok()
        .and_then(|value| value.parse::<u64>().ok())
        .filter(|days| (1..=90).contains(days))
        .unwrap_or(DEFAULT_EXPIRY_DAYS);

    let default_expiry_ms = (now_secs + expiry_days * SECONDS_PER_DAY) * MILLIS_PER_SECOND;
    let expiry_ms = env::var("APP_EXPIRES_AT_EPOCH_MS")
        .ok()
        .and_then(|value| value.parse::<u64>().ok())
        .or_else(|| {
            env::var("APP_EXPIRES_AT")
                .ok()
                .and_then(|value| parse_rfc3339_z_ms(&value))
        })
        .unwrap_or(default_expiry_ms);

    let not_before_ms = env::var("APP_NOT_BEFORE_EPOCH_MS")
        .ok()
        .and_then(|value| value.parse::<u64>().ok())
        .unwrap_or_else(|| now_secs.saturating_sub(SECONDS_PER_DAY) * MILLIS_PER_SECOND);

    let out_dir = PathBuf::from(env::var("OUT_DIR").expect("OUT_DIR is set by cargo"));
    let build_info = format!(
        "const APP_EXPIRES_AT_EPOCH_MS: f64 = {expiry_ms}.0;\nconst APP_NOT_BEFORE_EPOCH_MS: f64 = {not_before_ms}.0;\n",
    );
    fs::write(out_dir.join("build_info.rs"), build_info).expect("write generated build info");
}

fn parse_rfc3339_z_ms(value: &str) -> Option<u64> {
    let bytes = value.as_bytes();
    if bytes.len() != 20
        || bytes[4] != b'-'
        || bytes[7] != b'-'
        || bytes[10] != b'T'
        || bytes[13] != b':'
        || bytes[16] != b':'
        || bytes[19] != b'Z'
    {
        return None;
    }

    let year = parse_i32(&value[0..4])?;
    let month = parse_u32(&value[5..7])?;
    let day = parse_u32(&value[8..10])?;
    let hour = parse_u32(&value[11..13])?;
    let minute = parse_u32(&value[14..16])?;
    let second = parse_u32(&value[17..19])?;

    if !(1..=12).contains(&month)
        || !(1..=31).contains(&day)
        || hour > 23
        || minute > 59
        || second > 59
    {
        return None;
    }

    let days = days_from_civil(year, month, day);
    if days < 0 {
        return None;
    }
    let seconds =
        days as u64 * SECONDS_PER_DAY + hour as u64 * 3_600 + minute as u64 * 60 + second as u64;
    Some(seconds * MILLIS_PER_SECOND)
}

fn parse_i32(value: &str) -> Option<i32> {
    value.parse::<i32>().ok()
}

fn parse_u32(value: &str) -> Option<u32> {
    value.parse::<u32>().ok()
}

fn days_from_civil(year: i32, month: u32, day: u32) -> i64 {
    let year = year - i32::from(month <= 2);
    let era = if year >= 0 { year } else { year - 399 } / 400;
    let yoe = year - era * 400;
    let month = month as i32;
    let day = day as i32;
    let doy = (153 * (month + if month > 2 { -3 } else { 9 }) + 2) / 5 + day - 1;
    let doe = yoe * 365 + yoe / 4 - yoe / 100 + doy;
    (era * 146_097 + doe - 719_468) as i64
}
