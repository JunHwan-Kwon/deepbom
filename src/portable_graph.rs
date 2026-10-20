//! Model IR port edges, boundary projection and connected bounded partitions.
//! Grouping never introduces connections from hierarchy or display position.
use serde_json::{json, Value};
use std::collections::{BTreeMap, BTreeSet, VecDeque};

fn rows<'a>(value: &'a Value, key: &str) -> Result<&'a Vec<Value>, String> {
    value[key]
        .as_array()
        .ok_or_else(|| format!("invalid graph {key}"))
}
fn text<'a>(value: &'a Value, key: &str) -> Result<&'a str, String> {
    value[key]
        .as_str()
        .ok_or_else(|| format!("invalid graph {key}"))
}
fn ids(values: &[Value]) -> Result<Vec<&str>, String> {
    values
        .iter()
        .map(|v| v.as_str().ok_or("invalid graph identity".into()))
        .collect()
}
fn unique(values: &[&str]) -> Result<(), String> {
    if values.iter().copied().collect::<BTreeSet<_>>().len() != values.len() {
        Err("Duplicate graph identity".into())
    } else {
        Ok(())
    }
}
fn components(members: &[usize], adjacent: &[BTreeSet<usize>]) -> Vec<Vec<usize>> {
    let included: BTreeSet<_> = members.iter().copied().collect();
    let mut component = BTreeMap::new();
    let mut output = vec![];
    for &member in members {
        if component.contains_key(&member) {
            continue;
        }
        let group = output.len();
        output.push(vec![]);
        component.insert(member, group);
        let mut queue = VecDeque::from([member]);
        while let Some(node) = queue.pop_front() {
            for next in &adjacent[node] {
                if included.contains(next) && !component.contains_key(next) {
                    component.insert(*next, group);
                    queue.push_back(*next);
                }
            }
        }
    }
    for member in members {
        output[component[member]].push(*member);
    }
    output
}
pub(crate) fn partition(input: &Value) -> Result<Value, String> {
    let members = ids(rows(input, "members")?)?;
    unique(&members)?;
    let limit = input["limit"]
        .as_u64()
        .filter(|v| *v > 0)
        .ok_or("Invalid structural group limit")? as usize;
    if limit == 0 {
        return Err("Invalid structural group limit".into());
    }
    let positions: BTreeMap<_, _> = members.iter().enumerate().map(|(i, id)| (*id, i)).collect();
    let mut adjacent = vec![BTreeSet::new(); members.len()];
    for edge in rows(input, "edges")? {
        if edge["kind"] != "data_dependency" {
            continue;
        }
        if let (Some(a), Some(b)) = (
            edge["from_ref"].as_str().and_then(|id| positions.get(id)),
            edge["to_ref"].as_str().and_then(|id| positions.get(id)),
        ) {
            adjacent[*a].insert(*b);
            adjacent[*b].insert(*a);
        }
    }
    let mut output = vec![];
    for component in components(&(0..members.len()).collect::<Vec<_>>(), &adjacent) {
        for chunk in component.chunks(limit) {
            output.extend(components(chunk, &adjacent));
        }
    }
    output.sort_unstable_by_key(|group| group[0]);
    Ok(json!(output
        .iter()
        .map(|group| group.iter().map(|i| members[*i]).collect::<Vec<_>>())
        .collect::<Vec<_>>()))
}
pub(crate) fn boundary(input: &Value) -> Result<Value, String> {
    let members = ids(rows(input, "members")?)?;
    unique(&members)?;
    let members: BTreeSet<_> = members.into_iter().collect();
    let operations: BTreeSet<_> = ids(rows(input, "operations")?)?.into_iter().collect();
    if !members.is_subset(&operations) {
        return Err("Group membership is duplicated or unresolved".into());
    }
    let mut incoming = vec![];
    let mut outgoing = vec![];
    let mut internal = vec![];
    let mut inputs = vec![];
    let mut outputs = vec![];
    let mut seen_in = BTreeSet::new();
    let mut seen_out = BTreeSet::new();
    for edge in rows(input, "edges")? {
        let from = edge["from_ref"]
            .as_str()
            .map(|id| members.contains(id))
            .unwrap_or(false);
        let to = edge["to_ref"]
            .as_str()
            .map(|id| members.contains(id))
            .unwrap_or(false);
        if !from && !to {
            continue;
        }
        let id = text(edge, "id")?;
        let value = text(edge, "value_ref")?;
        if from && to {
            internal.push(id);
        } else if to {
            incoming.push(id);
            if seen_in.insert(value) {
                inputs.push(value);
            }
        } else {
            outgoing.push(id);
            if seen_out.insert(value) {
                outputs.push(value);
            }
        }
    }
    Ok(
        json!({"input_value_refs":inputs,"output_value_refs":outputs,"internal_edge_refs":internal,"incoming_edge_refs":incoming,"outgoing_edge_refs":outgoing}),
    )
}
pub(crate) fn edges(input: &Value) -> Result<Value, String> {
    let ports = rows(input, "ports")?;
    let mut producers = BTreeMap::new();
    let mut consumers: BTreeMap<&str, Vec<&Value>> = BTreeMap::new();
    for port in ports {
        let value = text(port, "value_ref")?;
        if port["direction"] == "output" {
            if producers.insert(value, port).is_some() {
                return Err("Model value has multiple serialized producers".into());
            }
        } else {
            consumers.entry(value).or_default().push(port);
        }
    }
    let mut outputs: BTreeMap<&str, Vec<(&str, usize)>> = BTreeMap::new();
    for program in rows(input, "programs")? {
        let id = text(program, "id")?;
        if let Some(values) = program["output_value_refs"].as_array() {
            for (position, value) in ids(values)?.iter().enumerate() {
                outputs.entry(value).or_default().push((id, position));
            }
        }
    }
    let mut result = vec![];
    for value in rows(input, "values")? {
        let id = text(value, "id")?;
        let producer = producers.get(id);
        let from = producer.map(|p| &p["operation_ref"]);
        let port = producer.map(|p| &p["id"]);
        let position = producer.map(|p| &p["position"]);
        if let Some(rows) = consumers.get(id) {
            for consumer in rows {
                result.push(json!({"id":format!("port-edge:{}",text(consumer,"id")?),"value_ref":id,
                "from_ref":from,"to_ref":consumer["operation_ref"],"from_port_ref":port,"to_port_ref":consumer["id"],
                "from_port_position":position,"to_port_position":consumer["position"],"boundary_program_ref":null,"boundary_position":null,
                "kind":if producer.is_some(){"data_dependency"}else if value["storage_refs"].as_array().map(|a|!a.is_empty()).unwrap_or(false){"storage_input"}else{"external_input"}}));
            }
        }
        if let Some(rows) = outputs.get(id) {
            for (program, output_position) in rows {
                result.push(json!({"id":format!("output-edge:{program}:{output_position}:{id}"),"value_ref":id,
                "from_ref":from,"to_ref":null,"from_port_ref":port,"to_port_ref":null,"kind":"model_output",
                "from_port_position":position,"to_port_position":null,"boundary_program_ref":program,"boundary_position":output_position}));
            }
        }
    }
    Ok(json!(result))
}
pub(crate) fn topology(input: &Value) -> Result<Value, String> {
    let source = rows(input, "operations")?;
    let mut positions = BTreeMap::new();
    let mut rows: Vec<&Value> = vec![];
    for row in source {
        let index = row["index"].as_i64().ok_or("invalid operation index")?;
        if let Some(&position) = positions.get(&index) {
            rows[position] = row;
        } else {
            positions.insert(index, rows.len());
            rows.push(row);
        }
    }
    let mut producers = BTreeMap::new();
    let mut consumers = BTreeMap::new();
    let indices =
        |row: &Value, key: &str| -> Result<Vec<u64>, String> { rows_of_indices(row, key) };
    for (position, row) in rows.iter().enumerate() {
        for tensor in indices(row, "outputs")? {
            producers.entry(tensor).or_insert(position);
        }
        for tensor in indices(row, "inputs")? {
            *consumers.entry(tensor).or_insert(0usize) += 1;
        }
    }
    let mut predecessors = vec![vec![]; rows.len()];
    for (position, row) in rows.iter().enumerate() {
        let mut seen = BTreeSet::new();
        for tensor in indices(row, "inputs")? {
            if let Some(&producer) = producers.get(&tensor) {
                if producer != position && seen.insert(producer) {
                    predecessors[position].push(producer);
                }
            }
        }
    }
    let mut depths = vec![None; rows.len()];
    let mut visiting = vec![false; rows.len()];
    let mut cycle = false;
    // Explicit DFS stack preserves the existing cycle diagnostic/depth order
    // while avoiding a native or JS call-stack overflow on long serial graphs.
    for start in 0..rows.len() {
        if depths[start].is_some() {
            continue;
        }
        let mut stack = vec![(start, 0usize, 0usize)];
        visiting[start] = true;
        while let Some((node, next, maximum)) = stack.last_mut() {
            if *next < predecessors[*node].len() {
                let child = predecessors[*node][*next];
                if let Some(depth) = depths[child] {
                    *maximum = (*maximum).max(depth);
                    *next += 1;
                } else if visiting[child] {
                    cycle = true;
                    *next += 1;
                } else {
                    visiting[child] = true;
                    stack.push((child, 0, 0));
                }
            } else {
                depths[*node] = Some(if predecessors[*node].is_empty() {
                    0
                } else {
                    *maximum + 1
                });
                visiting[*node] = false;
                stack.pop();
            }
        }
    }
    let mut annotations = vec![];
    for (position, row) in rows.iter().enumerate() {
        let fan = indices(row, "outputs")?
            .iter()
            .map(|v| consumers.get(v).copied().unwrap_or(0))
            .max()
            .unwrap_or(0);
        let name = text(row, "name")?;
        let role = if [
            "QUANTIZE",
            "DEQUANTIZE",
            "QUANTIZELINEAR",
            "DEQUANTIZELINEAR",
            "DYNAMICQUANTIZELINEAR",
        ]
        .contains(&name)
        {
            "quant-boundary"
        } else if predecessors[position].len() > 1 {
            "branch-merge"
        } else if fan > 1 {
            "branch-split"
        } else {
            "through"
        };
        annotations.push(json!({"op_index":row["index"],"role":role,"depth":depths[position],"fan_out_max":fan,"predecessor_count":predecessors[position].len()}));
    }
    annotations.sort_unstable_by_key(|row| row["op_index"].as_i64());
    Ok(
        json!({"schema":"deepbom.graph_topology.v1","status":if cycle{"invalid_cycle_detected"}else{"assessed"},"evidence_class":"DERIVED",
        "method":"Derive producer/consumer edges from serialized tensor indices; compute longest predecessor depth, maximum output-tensor fan-out, and branch/quant-boundary roles.",
        "cycle_detected":cycle,"annotations":annotations}),
    )
}
fn rows_of_indices(value: &Value, key: &str) -> Result<Vec<u64>, String> {
    rows(value, key)?
        .iter()
        .map(|v| v.as_u64().ok_or("invalid tensor index".into()))
        .collect()
}
#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn partitions_preserve_disconnected_and_skip_edges() {
        let value = json!({"members":["a","b","c","d"],"limit":2,"edges":[{"kind":"data_dependency","from_ref":"a","to_ref":"c"},{"kind":"data_dependency","from_ref":"c","to_ref":"d"}]});
        assert_eq!(
            partition(&value).unwrap(),
            json!([["a", "c"], ["b"], ["d"]])
        );
    }
    #[test]
    fn boundary_preserves_ports_and_deduplicates_values() {
        let value = json!({"members":["b"],"operations":["a","b"],"edges":[{"id":"e0","from_ref":"a","to_ref":"b","value_ref":"x"},{"id":"e1","from_ref":"a","to_ref":"b","value_ref":"x"},{"id":"o","from_ref":"b","to_ref":null,"value_ref":"y"}]});
        let result = boundary(&value).unwrap();
        assert_eq!(result["input_value_refs"], json!(["x"]));
        assert_eq!(result["incoming_edge_refs"], json!(["e0", "e1"]));
        assert_eq!(result["outgoing_edge_refs"], json!(["o"]));
    }
    #[test]
    fn deep_and_cyclic_topology() {
        let ops: Vec<_> = (0..20000)
            .rev()
            .map(|i| json!({"index":i,"name":"CONV","inputs":[i],"outputs":[i+1]}))
            .collect();
        let graph = topology(&json!({"operations":ops})).unwrap();
        assert_eq!(graph["annotations"][19999]["depth"], 19999);
        let graph=topology(&json!({"operations":[{"index":0,"name":"ADD","inputs":[2],"outputs":[1]},{"index":1,"name":"ADD","inputs":[1],"outputs":[2]}]})).unwrap();
        assert_eq!(graph["cycle_detected"], true);
        assert_eq!(graph["annotations"][0]["depth"], 2);
    }
}
