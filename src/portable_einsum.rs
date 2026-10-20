//! ONNX equation grammar shared by static shape, symbolic cost and MAC readers.
use serde_json::{json, Value};
use std::collections::{BTreeMap, BTreeSet};
fn fail(reason: impl Into<String>) -> Value {
    json!({"status":"invalid","reason":reason.into(),"operands":[],"output":[],"ellipsis_labels":[],"reduction_labels":[],"all_labels":[]})
}
fn tokens(text: &str) -> Option<(Vec<String>, bool, usize)> {
    let mut rest = text;
    let mut out = vec![];
    let mut ellipsis = false;
    let mut labels = 0;
    while !rest.is_empty() {
        if let Some(next) = rest.strip_prefix("...") {
            if ellipsis {
                return None;
            }
            ellipsis = true;
            out.push("...".into());
            rest = next;
        } else {
            if !rest.as_bytes()[0].is_ascii_alphabetic() {
                return None;
            }
            out.push(rest[..1].into());
            rest = &rest[1..];
            labels += 1;
        }
    }
    Some((out, ellipsis, labels))
}
pub(crate) fn parse(v: &Value) -> Value {
    let source = v["source"].as_str().unwrap_or("");
    if source.is_empty() {
        return fail("einsum_equation_missing");
    }
    let parts: Vec<_> = source.split("->").collect();
    if parts.len() > 2 {
        return fail("einsum_equation_multiple_output_arrows");
    }
    let input_texts: Vec<_> = parts[0].split(',').collect();
    let Some(ranks) = v["ranks"].as_array() else {
        return fail("einsum_equation_input_cardinality_mismatch");
    };
    if input_texts.len() != ranks.len() {
        return fail("einsum_equation_input_cardinality_mismatch");
    }
    let mut parsed = vec![];
    let mut max_rank = 0;
    for (i, text) in input_texts.iter().enumerate() {
        let Some((tokens, ellipsis, labels)) = tokens(text) else {
            return fail(format!("einsum_input_subscript_invalid:{i}"));
        };
        let Some(rank) = ranks[i].as_u64().filter(|n| *n <= 9_007_199_254_740_991) else {
            return fail(format!("einsum_input_rank_invalid:{i}"));
        };
        if rank < labels as u64 || !ellipsis && rank != labels as u64 {
            return fail(format!("einsum_input_rank_subscript_mismatch:{i}"));
        }
        // Bounds before expansion; a malicious rank must not exhaust WASM.
        if rank > 100_000 {
            return fail(format!("einsum_input_rank_budget_exceeded:{i}"));
        }
        let er = rank as usize - labels;
        max_rank = max_rank.max(er);
        parsed.push((tokens, er));
    }
    if max_rank.saturating_mul(parsed.len()) > 1_000_000 {
        return fail("einsum_expansion_budget_exceeded");
    }
    let ellipsis: Vec<_> = (0..max_rank).map(|i| format!("@ellipsis:{i}")).collect();
    let operands: Vec<Vec<String>> = parsed
        .into_iter()
        .map(|(tokens, rank)| {
            tokens
                .into_iter()
                .flat_map(|t| {
                    if t == "..." {
                        ellipsis[max_rank - rank..].to_vec()
                    } else {
                        vec![t]
                    }
                })
                .collect()
        })
        .collect();
    let all: BTreeSet<_> = operands.iter().flatten().cloned().collect();
    let output = if parts.len() == 2 {
        let Some((tokens, _, _)) = tokens(parts[1]) else {
            return fail("einsum_output_subscript_invalid");
        };
        let output: Vec<_> = tokens
            .into_iter()
            .flat_map(|t| {
                if t == "..." {
                    ellipsis.clone()
                } else {
                    vec![t]
                }
            })
            .collect();
        if output.iter().collect::<BTreeSet<_>>().len() != output.len() {
            return fail("einsum_output_label_duplicate");
        }
        if output.iter().any(|label| !all.contains(label)) {
            return fail("einsum_output_label_absent_from_inputs");
        }
        output
    } else {
        let mut counts = BTreeMap::new();
        for label in operands
            .iter()
            .flatten()
            .filter(|l| !l.starts_with("@ellipsis:"))
        {
            *counts.entry(label.clone()).or_insert(0) += 1;
        }
        ellipsis
            .iter()
            .cloned()
            .chain(counts.into_iter().filter(|(_, n)| *n == 1).map(|(l, _)| l))
            .collect()
    };
    let reductions: Vec<_> = all
        .iter()
        .filter(|l| !output.contains(l))
        .cloned()
        .collect();
    json!({"status":"assessed","operands":operands,"output":output,"ellipsis_labels":ellipsis,"reduction_labels":reductions,"all_labels":all,"explicit_output":parts.len()==2})
}
#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn implicit_and_explicit_broadcast() {
        let result = parse(&json!({"source":"...ij,...jk->...ik","ranks":[4,3]}));
        assert_eq!(
            result["output"],
            json!(["@ellipsis:0", "@ellipsis:1", "i", "k"])
        );
        assert_eq!(result["operands"][1], json!(["@ellipsis:1", "j", "k"]));
        assert_eq!(
            parse(&json!({"source":"ji","ranks":[2]}))["output"],
            json!(["i", "j"])
        );
        assert_eq!(
            parse(&json!({"source":"ij->ii","ranks":[2]}))["reason"],
            "einsum_output_label_duplicate"
        );
        assert_eq!(
            parse(&json!({"source":"...","ranks":[1_000_000_000]}))["reason"],
            "einsum_input_rank_budget_exceeded:0"
        );
    }
}
