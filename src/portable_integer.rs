//! Exact guarded expressions used by dynamic-shape costs. Static contraction
//! costs call the same contributing-pair implementation as symbolic consumers.
use num_bigint::BigInt;
use num_traits::{One, ToPrimitive, Zero};
use serde_json::{json, Value};

const MAX_BITS: u64 = 131072;
fn integer(v: &Value) -> Option<BigInt> {
    let s = v.as_str()?;
    let digits = s.strip_prefix('-').unwrap_or(s);
    if digits.is_empty()
        || digits.len() > 32768
        || (digits.len() > 1 && digits.starts_with('0'))
        || !digits.bytes().all(|b| b.is_ascii_digit())
    {
        return None;
    }
    s.parse().ok()
}
fn floor_div(a: BigInt, b: &BigInt) -> BigInt {
    let q = &a / b;
    if &a % b < BigInt::zero() {
        q - 1
    } else {
        q
    }
}
// Shared finite arithmetic for guarded cost expressions and dimension formulas.
fn arithmetic(op: &str, args: &[BigInt]) -> Option<BigInt> {
    match op {
        "add" => Some(args.iter().sum()),
        "mul" => {
            let mut n = BigInt::one();
            for a in args {
                n *= a;
                if n.bits() > MAX_BITS {
                    return None;
                }
            }
            Some(n)
        }
        "sub" if args.len() == 2 => Some(&args[0] - &args[1]),
        "div" if args.len() == 2 && !args[1].is_zero() => Some(&args[0] / &args[1]),
        "reshape_quotient"
            if args.len() == 2 && !args[1].is_zero() && (&args[0] % &args[1]).is_zero() =>
        {
            Some(&args[0] / &args[1])
        }
        "exact_div"
            if args.len() == 2 && args[1] > BigInt::zero() && (&args[0] % &args[1]).is_zero() =>
        {
            Some(&args[0] / &args[1])
        }
        "floor_div" if args.len() == 2 && args[1] > BigInt::zero() => {
            Some(floor_div(args[0].clone(), &args[1]))
        }
        "ceil_div" if args.len() == 2 && args[1] > BigInt::zero() => {
            Some(-floor_div(-&args[0], &args[1]))
        }
        "max" => args.iter().max().cloned(),
        "min" => args.iter().min().cloned(),
        "broadcast_dim" => {
            if args.iter().any(|a| a < &BigInt::zero()) {
                return None;
            }
            let mut size = BigInt::one();
            for a in args {
                if !a.is_one() {
                    if !size.is_one() && size != *a {
                        return None;
                    }
                    size = a.clone();
                }
            }
            Some(size)
        }
        "range_len" if args.len() == 3 && !args[2].is_zero() => {
            let n = if args[2] > BigInt::zero() {
                -floor_div(&args[0] - &args[1], &args[2])
            } else {
                -floor_div(&args[1] - &args[0], &(-&args[2]))
            };
            Some(n.max(BigInt::zero()))
        }
        "conv_transpose_pairs" => pairs(args),
        _ => None,
    }
}
#[derive(Clone)]
pub(crate) enum Bound {
    Finite(BigInt),
    Infinity(i8),
}
pub(crate) fn slice_range(
    size: &BigInt,
    start: &Bound,
    end: &Bound,
    step: &BigInt,
) -> Option<(BigInt, BigInt, BigInt)> {
    if size < &BigInt::zero() || step.is_zero() {
        return None;
    }
    let positive = step > &BigInt::zero();
    let low = if positive {
        BigInt::zero()
    } else {
        -BigInt::one()
    };
    let high = if positive { size.clone() } else { size - 1 };
    let clamp = |v: &Bound| match v {
        Bound::Infinity(sign) => {
            if *sign < 0 {
                low.clone()
            } else {
                high.clone()
            }
        }
        Bound::Finite(n) => {
            let x = if n < &BigInt::zero() {
                n + size
            } else {
                n.clone()
            };
            x.max(low.clone()).min(high.clone())
        }
    };
    let a = clamp(start);
    let b = clamp(end);
    let length = if positive {
        -floor_div(&a - &b, step)
    } else {
        -floor_div(&b - &a, &(-step))
    }
    .max(BigInt::zero());
    Some((a, b, length))
}
fn dimension(node: &Value, assignments: &Value, budget: &mut usize) -> Option<Bound> {
    *budget = budget.checked_sub(1)?;
    let n = match node["kind"].as_str()? {
        "infinity" => {
            return Some(Bound::Infinity(if node["sign"].as_i64()? < 0 {
                -1
            } else {
                1
            }))
        }
        "constant" => integer(&node["value_decimal"]),
        "symbol" => integer(&assignments[node["name"].as_str()?]),
        "call" => {
            let args = node["arguments"]
                .as_array()?
                .iter()
                .map(|n| dimension(n, assignments, budget))
                .collect::<Option<Vec<_>>>()?;
            let op = node["operator"].as_str()?;
            if op == "slice_len" {
                let [Bound::Finite(size), a, b, Bound::Finite(step)] = args.as_slice() else {
                    return None;
                };
                Some(slice_range(size, a, b, step)?.2)
            } else {
                let args = args
                    .into_iter()
                    .map(|a| {
                        if let Bound::Finite(n) = a {
                            Some(n)
                        } else {
                            None
                        }
                    })
                    .collect::<Option<Vec<_>>>()?;
                arithmetic(op, &args)
            }
        }
        _ => None,
    }?;
    (n.bits() <= MAX_BITS).then_some(Bound::Finite(n))
}
pub(crate) fn evaluate_dimension(input: &Value) -> Value {
    let mut budget = 100000;
    match dimension(&input["expression"], &input["assignments"], &mut budget) {
        Some(Bound::Finite(n)) if n >= BigInt::zero() => json!(n.to_string()),
        _ => Value::Null,
    }
}
pub(crate) fn pairs(values: &[BigInt]) -> Option<BigInt> {
    let [input, kernel, stride, dilation, pad, output] = values else {
        return None;
    };
    let zero = BigInt::zero();
    if [input, pad, output].iter().any(|v| **v < zero)
        || [kernel, stride, dilation].iter().any(|v| **v <= zero)
    {
        return None;
    }
    let (count, step, bound, denom) = if kernel <= input {
        (kernel.to_u64()?, dilation, input, stride)
    } else {
        (input.to_u64()?, stride, kernel, dilation)
    };
    if count > 1_000_000 || values.iter().any(|v| v.bits() > MAX_BITS) {
        return None;
    }
    let mut total = BigInt::zero();
    let mut offset = BigInt::zero();
    let minus_one = -BigInt::one();
    let last: BigInt = bound - 1;
    for _ in 0..count {
        let low = floor_div(pad - &offset - 1, denom).max(minus_one.clone()) + 1;
        let high = floor_div(output - 1 + pad - &offset, denom).min(last.clone());
        if high >= low {
            total += high - low + 1;
        }
        offset += step;
    }
    Some(total)
}
fn expression(node: &Value, assignments: &Value, budget: &mut usize) -> Option<BigInt> {
    *budget = budget.checked_sub(1)?;
    let result = match node["kind"].as_str()? {
        "constant" => integer(&node["value_decimal"]),
        "symbol" => {
            integer(&assignments[node["symbol_id"].as_str()?]).filter(|n| n >= &BigInt::zero())
        }
        "call" => {
            let args: Vec<_> = node["arguments"]
                .as_array()?
                .iter()
                .map(|n| expression(n, assignments, budget))
                .collect::<Option<_>>()?;
            let op = node["operator"].as_str()?;
            if ["add", "mul", "sub", "exact_div", "conv_transpose_pairs"].contains(&op) {
                arithmetic(op, &args)
            } else {
                None
            }
        }
        _ => None,
    }?;
    (result.bits() <= MAX_BITS).then_some(result)
}
fn guard(value: &Value, assignments: &Value, budget: &mut usize) -> bool {
    match value["kind"].as_str() {
        Some("nonnegative") => expression(&value["expression_ir"], assignments, budget)
            .is_some_and(|n| n >= BigInt::zero()),
        Some("divisible") => match (
            expression(&value["numerator_ir"], assignments, budget),
            expression(&value["denominator_ir"], assignments, budget),
        ) {
            (Some(a), Some(b)) => {
                a >= BigInt::zero() && b > BigInt::zero() && a % b == BigInt::zero()
            }
            _ => false,
        },
        Some("equal") => match (
            expression(&value["left_ir"], assignments, budget),
            expression(&value["right_ir"], assignments, budget),
        ) {
            (Some(a), Some(b)) => a == b,
            _ => false,
        },
        _ => false,
    }
}
pub(crate) fn evaluate(input: &Value) -> Value {
    let mut budget = 100000;
    let assignments = &input["assignments"];
    if let Some(formula) = input.get("formula") {
        if formula["status"] != "exact_guarded_integer_expression"
            || formula["ir_schema"] != "deepbom.guarded_integer_expression.v2"
        {
            return Value::Null;
        }
        if let Some(guards) = formula.get("preconditions") {
            let Some(guards) = guards.as_array() else {
                return Value::Null;
            };
            if !guards.iter().all(|g| guard(g, assignments, &mut budget)) {
                return Value::Null;
            }
        }
        json!(
            expression(&formula["expression_ir"], assignments, &mut budget)
                .filter(|n| n >= &BigInt::zero())
                .map(|n| n.to_string())
        )
    } else {
        json!(expression(&input["expression"], assignments, &mut budget).map(|n| n.to_string()))
    }
}
#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn exact_guarded_expression() {
        let c = |s: &str| json!({"kind":"constant","value_decimal":s});
        let expr = json!({"kind":"call","operator":"mul","arguments":[c("9007199254740993"), c("9007199254740993")]});
        assert_eq!(
            evaluate(&json!({"expression":expr})),
            "81129638414606699710187514626049"
        );
        assert_eq!(
            evaluate(
                &json!({"expression":{"kind":"call","operator":"exact_div","arguments":[c("-6"),c("2")]}})
            ),
            "-3"
        );
        assert!(evaluate(
            &json!({"expression":{"kind":"symbol","symbol_id":"n"},"assignments":{"n":"-1"}})
        )
        .is_null());
    }
    #[test]
    fn large_pairs_and_budget() {
        let values = [
            "10000000000000000000000000000000000000000",
            "1",
            "1",
            "1",
            "0",
            "10000000000000000000000000000000000000000",
        ];
        assert_eq!(
            pairs(&values.map(|s| s.parse().unwrap()))
                .unwrap()
                .to_string(),
            values[0]
        );
        assert!(pairs(&[1000001, 1000001, 1, 1, 0, 1].map(BigInt::from)).is_none());
    }
}
