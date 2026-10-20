//! Executable scaffold admission. Code rendering remains a host adapter; graph,
//! port, cardinality and spatial checks use the shared portable rule owners.
use num_bigint::BigUint;
use serde_json::{json, Value};
use std::collections::{BTreeMap, BTreeSet};

const SAFE: u64 = 9_007_199_254_740_991;
fn index(v: &Value) -> Option<u64> {
    v.as_u64().filter(|n| *n <= SAFE)
}
fn positive(v: &Value) -> Option<u64> {
    index(v).filter(|n| *n > 0)
}
fn count(v: &Value) -> Option<BigUint> {
    let a = v.as_array()?;
    if a.is_empty() {
        return None;
    }
    if a.iter().any(|d| positive(d).is_none()) {
        return None;
    }
    crate::portable_contracts::count_product(a).ok().flatten()
}
fn spatial(node: &Value) -> Result<(), &'static str> {
    let s = &node["spatial_options"];
    let conv = matches!(
        node["module_kind"].as_str(),
        Some("conv2d" | "depthwise_conv2d")
    );
    if !matches!(s["padding"].as_str(), Some("same" | "valid"))
        || ![&s["strides"], &s["dilations"]].iter().all(|a| {
            a.as_array()
                .is_some_and(|a| a.len() == 2 && a.iter().all(|n| positive(n).is_some()))
        })
    {
        return Err("Invalid serialized spatial options");
    }
    let stride = s["strides"].as_array().unwrap();
    let dilation = s["dilations"].as_array().unwrap();
    if conv
        && stride.iter().any(|n| positive(n).unwrap() > 1)
        && dilation.iter().any(|n| positive(n).unwrap() > 1)
    {
        return Err(
            "Combined stride and dilation are outside the qualified Keras scaffold contract",
        );
    }
    if conv && (node["stride_height"] != stride[0] || node["stride_width"] != stride[1]) {
        return Err("Convolution stride fields disagree");
    }
    let x = node["input_shapes"][0]
        .as_array()
        .ok_or("Spatial operator requires rank four")?;
    let y = node["output_shapes"][0]
        .as_array()
        .ok_or("Spatial operator requires rank four")?;
    if x.len() != 4 || y.len() != 4 {
        return Err("Spatial operator requires rank four");
    }
    let attrs = json!({"kernel_shape":[node["kernel_height"],node["kernel_width"]],"strides":stride,"dilations":if conv{json!(dilation)}else{json!([1,1])},"auto_pad":if s["padding"]=="same"{"SAME_UPPER"}else{"VALID"}});
    let out = if conv {
        let groups = if node["module_kind"] == "depthwise_conv2d" {
            positive(&node["input_channels"])
        } else {
            node.get("groups")
                .filter(|v| !v.is_null())
                .and_then(positive)
                .or_else(|| {
                    if node.get("groups").is_none_or(Value::is_null) {
                        Some(1)
                    } else {
                        None
                    }
                })
        }
        .ok_or("Invalid convolution groups")?;
        let mut attrs = attrs;
        attrs["group"] = json!(groups);
        crate::portable_shape::run(&json!({"operation":"conv","input":[x[0],x[3],x[1],x[2]],"weight":[node["output_channels"],positive(&node["input_channels"]).ok_or("Invalid input channels")?/groups,node["kernel_height"],node["kernel_width"]],"attributes":attrs})).map_err(|_|"Projected spatial output disagrees with the common shape rule")?
    } else {
        crate::portable_shape::run(&json!({"operation":"spatial","input":[x[1],x[2]],"kernels":[node["kernel_height"],node["kernel_width"]],"attributes":attrs})).map_err(|_|"Projected spatial output disagrees with the common shape rule")?
    };
    let values: Option<Vec<_>> = out.as_array().and_then(|a| {
        a.iter()
            .map(|d| {
                if d["kind"] == "value" {
                    Some(d["value"].clone())
                } else {
                    None
                }
            })
            .collect()
    });
    let expected = if conv {
        vec![y[0].clone(), y[3].clone(), y[1].clone(), y[2].clone()]
    } else {
        vec![y[1].clone(), y[2].clone()]
    };
    if values != Some(expected) || (!conv && (x[0] != y[0] || x[3] != y[3])) {
        return Err("Projected spatial output disagrees with the common shape rule");
    }
    Ok(())
}
fn validate(plan: &Value, executable: bool) -> Result<(), &'static str> {
    if plan["schema"] != "deepbom.redesign_implementation_plan.v1" {
        return Err("Invalid redesign implementation plan.");
    }
    let nodes = plan["nodes"]
        .as_array()
        .ok_or("Invalid redesign implementation plan.")?;
    let mut shapes = BTreeMap::new();
    let mut ids = BTreeSet::new();
    let mut symbols = BTreeSet::new();
    let empty = vec![];
    for input in plan
        .get("model_inputs")
        .unwrap_or(&Value::Null)
        .as_array()
        .unwrap_or(&empty)
    {
        let id = index(&input["tensor_index"]).ok_or("Invalid model input identity.")?;
        if shapes.insert(id, &input["projected_shape"]).is_some() {
            return Err("Implementation input has duplicate identity.");
        }
    }
    for node in nodes {
        let id = index(&node["op_index"])
            .ok_or("Invalid or duplicate implementation operation identity.")?;
        let symbol = node["generated_symbol"].as_str().unwrap_or("");
        if !ids.insert(id)
            || !symbol
                .strip_prefix("op_")
                .is_some_and(|s| !s.is_empty() && s.bytes().all(|c| c.is_ascii_digit()))
            || !symbols.insert(symbol)
        {
            return Err("Invalid or duplicate implementation operation identity.");
        }
        let ins = node["activation_inputs"]
            .as_array()
            .ok_or("Implementation ports and shape inventory disagree.")?;
        let outs = node["outputs"]
            .as_array()
            .ok_or("Implementation ports and shape inventory disagree.")?;
        let xs = node["input_shapes"]
            .as_array()
            .ok_or("Implementation ports and shape inventory disagree.")?;
        let ys = node["output_shapes"]
            .as_array()
            .ok_or("Implementation ports and shape inventory disagree.")?;
        if ins.len() != xs.len() || outs.len() != ys.len() {
            return Err("Implementation ports and shape inventory disagree.");
        }
        for (i, port) in ins.iter().enumerate() {
            if index(port).and_then(|id| shapes.get(&id).copied()) != Some(&xs[i]) {
                return Err(
                    "Implementation input is unavailable or its shape disagrees with the producer.",
                );
            }
        }
        for (i, port) in outs.iter().enumerate() {
            let id = index(port).ok_or("Invalid implementation output identity.")?;
            if shapes.insert(id, &ys[i]).is_some() {
                return Err("Implementation output has multiple producers.");
            }
        }
        if !executable {
            continue;
        }
        if node["codegen_status"] == "unsupported"
            || ys.len() != 1
            || xs.is_empty()
            || xs.iter().chain(ys).any(|s| count(s).is_none())
        {
            return Err("Implementation has unsupported or incomplete executable contracts.");
        }
        let kind = node["module_kind"].as_str().unwrap_or("");
        let binary = matches!(
            kind,
            "add" | "subtract" | "multiply" | "maximum" | "minimum"
        );
        let unary = matches!(
            kind,
            "conv2d"
                | "depthwise_conv2d"
                | "linear"
                | "average_pool2d"
                | "max_pool2d"
                | "reshape"
                | "squeeze"
                | "relu"
                | "relu6"
                | "sigmoid"
                | "softmax"
                | "precision_boundary"
        );
        if !binary && !unary && kind != "concatenate" {
            return Err("Implementation operation has no reviewed framework emitter.");
        }
        if (binary && xs.len() != 2) || (unary && xs.len() != 1) {
            return Err("Implementation input arity is not supported by both framework emitters.");
        }
        if kind == "concatenate" {
            let actual = crate::portable_shape::run(
                &json!({"operation":"concat","shapes":xs,"axis":node["concatenation_axis_nhwc"]}),
            )
            .map_err(|_| "Concatenation axis or output shape is invalid.")?;
            if actual
                .as_array()
                .map(|a| a.iter().map(|d| d["value"].clone()).collect::<Vec<_>>())
                .as_ref()
                != ys[0].as_array()
            {
                return Err("Concatenation axis or output shape is invalid.");
            }
        }
        if matches!(kind, "reshape" | "squeeze")
            && (count(&xs[0]) != count(&ys[0]) || xs[0][0] != ys[0][0])
        {
            return Err("Reshape must conserve elements and the batch dimension.");
        }
        if matches!(kind, "conv2d" | "depthwise_conv2d") {
            let groups = if kind == "depthwise_conv2d" {
                node["input_channels"].clone()
            } else {
                node.get("groups")
                    .filter(|v| !v.is_null())
                    .cloned()
                    .unwrap_or(json!(1))
            };
            let valid = [
                &node["input_channels"],
                &node["output_channels"],
                &node["kernel_height"],
                &node["kernel_width"],
                &groups,
            ]
            .iter()
            .all(|n| positive(n).is_some());
            if !valid
                || positive(&node["input_channels"]).unwrap() % positive(&groups).unwrap() != 0
                || positive(&node["output_channels"]).unwrap() % positive(&groups).unwrap() != 0
                || xs[0].as_array().unwrap().len() != 4
                || ys[0].as_array().unwrap().len() != 4
                || xs[0][3] != node["input_channels"]
                || ys[0][3] != node["output_channels"]
            {
                return Err(
                    "Convolution groups, channels or kernel contradict the tensor contract.",
                );
            }
        }
        let options = node.get("spatial_options").filter(|v| !v.is_null());
        if matches!(
            kind,
            "conv2d" | "depthwise_conv2d" | "average_pool2d" | "max_pool2d"
        ) && options.is_some()
        {
            spatial(node)?;
        }
        if kind == "linear" && (ys[0].as_array().unwrap().len() != 2 || ys[0][0] != xs[0][0]) {
            return Err("Linear scaffold requires a rank-2 output and preserved batch.");
        }
        if matches!(
            kind,
            "relu" | "relu6" | "sigmoid" | "softmax" | "precision_boundary"
        ) && xs[0] != ys[0]
        {
            return Err("Elementwise unary operation changed its shape.");
        }
        if matches!(
            kind,
            "add" | "subtract" | "multiply" | "maximum" | "minimum"
        ) {
            if xs
                .iter()
                .any(|x| x.as_array().unwrap().len() != ys[0].as_array().unwrap().len())
            {
                return Err("Mixed-rank broadcasting is not qualified by both emitters.");
            }
            let actual = crate::portable_shape::run(&json!({"operation":"broadcast","shapes":xs}))
                .map_err(|_| "Elementwise broadcast disagrees with the output shape.")?;
            if actual
                .as_array()
                .map(|a| a.iter().map(|d| d["value"].clone()).collect::<Vec<_>>())
                .as_ref()
                != ys[0].as_array()
            {
                return Err("Elementwise broadcast disagrees with the output shape.");
            }
        }
        if matches!(kind, "average_pool2d" | "max_pool2d")
            && options.is_none()
            && (ys[0].as_array().unwrap().len() != 4 || ys[0][1] != 1 || ys[0][2] != 1)
        {
            return Err("Legacy pooling plan lacks serialized window/stride/padding; regenerate the plan from the artifact.");
        }
        if let Some(s) = options {
            if !matches!(s["padding"].as_str(), Some("same" | "valid"))
                || ![&s["strides"], &s["dilations"]].iter().all(|a| {
                    a.as_array()
                        .is_some_and(|a| a.len() == 2 && a.iter().all(|n| positive(n).is_some()))
                })
            {
                return Err("Invalid serialized spatial options.");
            }
            if s["strides"]
                .as_array()
                .unwrap()
                .iter()
                .any(|n| positive(n).unwrap() > 1)
                && s["dilations"]
                    .as_array()
                    .unwrap()
                    .iter()
                    .any(|n| positive(n).unwrap() > 1)
            {
                return Err(
                    "This stride/dilation combination is not supported by both framework emitters.",
                );
            }
        }
        if let Some(s) = options {
            if matches!(kind, "average_pool2d" | "max_pool2d")
                && s["dilations"].as_array().unwrap().iter().any(|n| *n != 1)
            {
                return Err("Pooling dilation is not supported by both framework emitters.");
            }
        }
        if !["NONE", "RELU", "RELU6", "TANH"].contains(
            &node["fused_activation"]
                .as_str()
                .filter(|s| !s.is_empty())
                .unwrap_or("NONE")
                .to_uppercase()
                .as_str(),
        ) {
            return Err("Unsupported fused activation in implementation plan.");
        }
    }
    for output in plan
        .get("model_outputs")
        .unwrap_or(&Value::Null)
        .as_array()
        .unwrap_or(&empty)
    {
        if index(&output["tensor_index"]).and_then(|id| shapes.get(&id).copied())
            != Some(&output["projected_shape"])
        {
            return Err("Implementation output contract disagrees with the graph.");
        }
    }
    Ok(())
}
pub(crate) fn run(input: &Value) -> Value {
    json!(validate(
        &input["plan"],
        input["executable"].as_bool().unwrap_or(false)
    )
    .err())
}
