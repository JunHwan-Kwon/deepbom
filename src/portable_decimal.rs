//! Decimal arithmetic on the ECMAScript spelling of a reported binary64 value.
//! This preserves reported precision; it never invents a confidence interval.
use num_bigint::BigInt;
use num_traits::{Signed, Zero};
use serde_json::{json, Value};
#[derive(Clone)]
struct Decimal {
    coefficient: BigInt,
    exponent: i32,
}
impl Decimal {
    fn input(v: &Value) -> Result<Self, String> {
        let s = v["coefficient"]
            .as_str()
            .ok_or("decimal coefficient must be text")?;
        let exponent = v["exponent"]
            .as_i64()
            .filter(|x| (-10000..=10000).contains(x))
            .ok_or("decimal exponent exceeds budget")? as i32;
        if s.len() > 32768 {
            return Err("decimal coefficient exceeds budget".into());
        }
        Ok(Self {
            coefficient: s.parse().map_err(|_| "invalid decimal coefficient")?,
            exponent,
        })
    }
    fn reported(v: &Value) -> Result<Self, String> {
        let f = v
            .as_f64()
            .filter(|x| x.is_finite())
            .ok_or("Reported decimal requires a finite number")?;
        let s = ryu_js::Buffer::new().format_finite(f).to_owned();
        let (base, exp) = s
            .split_once('e')
            .map(|(b, e)| (b, e.parse::<i32>().unwrap()))
            .unwrap_or((&s, 0));
        let fraction = base.split_once('.').map(|(_, f)| f.len()).unwrap_or(0);
        Ok(Self {
            coefficient: base
                .replace('.', "")
                .parse()
                .map_err(|_| "invalid reported decimal")?,
            exponent: exp - fraction as i32,
        })
    }
    fn add(&self, b: &Self) -> Self {
        let e = self.exponent.min(b.exponent);
        Self {
            coefficient: &self.coefficient * BigInt::from(10).pow((self.exponent - e) as u32)
                + &b.coefficient * BigInt::from(10).pow((b.exponent - e) as u32),
            exponent: e,
        }
    }
    fn negate(&self) -> Self {
        Self {
            coefficient: -&self.coefficient,
            exponent: self.exponent,
        }
    }
    fn value(&self) -> Value {
        json!({"coefficient":self.coefficient.to_string(),"exponent":self.exponent})
    }
    fn text(&self) -> String {
        if self.coefficient.is_zero() {
            return "0".into();
        }
        let digits = self.coefficient.abs().to_string();
        let point = digits.len() as i32 + self.exponent;
        let mut out = if point <= 0 {
            format!("0.{}{digits}", "0".repeat((-point) as usize))
        } else if point >= digits.len() as i32 {
            format!("{digits}{}", "0".repeat(point as usize - digits.len()))
        } else {
            format!(
                "{}.{}",
                &digits[..point as usize],
                &digits[point as usize..]
            )
        };
        if out.contains('.') {
            out = out.trim_end_matches('0').trim_end_matches('.').to_owned();
        }
        if self.coefficient.is_negative() {
            out.insert(0, '-');
        }
        out
    }
}
pub(crate) fn run(v: &Value) -> Result<Value, String> {
    let op = v["operation"].as_str().unwrap_or("");
    if op == "reported" {
        return Ok(Decimal::reported(&v["value"])?.value());
    }
    let a = Decimal::input(&v["a"])?;
    Ok(match op {
        "add" => a.add(&Decimal::input(&v["b"])?).value(),
        "subtract" => a.add(&Decimal::input(&v["b"])?.negate()).value(),
        "negate" => a.negate().value(),
        "halve" => Decimal {
            coefficient: a.coefficient * 5,
            exponent: a.exponent - 1,
        }
        .value(),
        "compare" => json!(
            match a.add(&Decimal::input(&v["b"])?.negate()).coefficient.sign() {
                num_bigint::Sign::Minus => -1,
                num_bigint::Sign::NoSign => 0,
                num_bigint::Sign::Plus => 1,
            }
        ),
        "text" => json!(a.text()),
        "number" => {
            let f = a
                .text()
                .parse::<f64>()
                .map_err(|_| "decimal conversion failed")?;
            if f.is_finite() && (f != 0. || a.coefficient.is_zero()) {
                json!(f)
            } else {
                Value::Null
            }
        }
        _ => return Err("unknown exact decimal operation".into()),
    })
}
#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn extremes() {
        let a = Decimal::reported(&json!(0.4)).unwrap();
        let b = Decimal::reported(&json!(0.1)).unwrap();
        assert_eq!(a.add(&b.negate()).text(), "0.3");
        assert_eq!(
            Decimal::reported(&json!(5e-324)).unwrap().text(),
            format!("0.{}5", "0".repeat(323))
        );
    }
}
