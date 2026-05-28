//! Interactive Plotly-based 3D scatter rendering. Saga 33
//! step 030: emits a self-contained HTML fragment (a `<div>`
//! plus an inline `<script>` calling `Plotly.newPlot`) instead
//! of an SVG. The web playground detects payloads starting
//! with the `<!-- mlpl-plotly3d -->` marker and renders them
//! as raw innerHTML in a sandboxed container; the embedded
//! script wires `plotly_click` events back to the REPL via
//! the global `window.mlpl_plotly_click` callback.
//!
//! `aux` may be a length-N integer or string label array,
//! splitting the data into one Plotly trace per unique label
//! (colored from the Catppuccin palette). The trace-to-sample
//! mapping is embedded in the HTML so the click handler can
//! recover the ORIGINAL sample index from
//! `(curveNumber, pointNumber)`.

use mlpl_array::DenseArray;
use std::fmt::Write;

use crate::svg::VizError;

const MARKER: &str = "<!-- mlpl-plotly3d -->";
const PALETTE: &[&str] = &[
    "#89b4fa", "#f38ba8", "#a6e3a1", "#fab387", "#cba6f7", "#94e2d5", "#f9e2af", "#eba0ac",
];

type Point3 = (f64, f64, f64);

/// Render `[N, 3]` points as an interactive Plotly 3D scatter.
/// `aux` is an optional length-N label array (integer or
/// integer-valued float); when present, points are split into
/// per-label traces with palette colors.
pub fn render_plotly3d(data: &DenseArray, aux: Option<&DenseArray>) -> Result<String, VizError> {
    let (n, points) = validate_plotly3d(data, aux)?;
    let labels = aux.map(|a| a.data().iter().map(|v| *v as i64).collect::<Vec<_>>());
    let uuid = stable_uuid(data, aux);
    let traces = build_traces(&points, labels.as_deref(), n);
    let map = build_cluster_map(labels.as_deref(), n);
    Ok(emit_html(&uuid, &traces, &map))
}

fn validate_plotly3d(
    data: &DenseArray,
    aux: Option<&DenseArray>,
) -> Result<(usize, Vec<Point3>), VizError> {
    let dims = data.shape().dims();
    if dims.len() != 2 || dims[1] != 3 {
        return Err(VizError::InvalidShape(format!(
            "plotly3d expects Nx3 points, got {dims:?}"
        )));
    }
    let n = dims[0];
    if let Some(a) = aux
        && a.data().len() != n
    {
        return Err(VizError::InvalidShape(format!(
            "plotly3d aux labels length {} must match {n} points",
            a.data().len()
        )));
    }
    let d = data.data();
    let points: Vec<(f64, f64, f64)> = (0..n)
        .map(|i| (d[i * 3], d[i * 3 + 1], d[i * 3 + 2]))
        .collect();
    Ok((n, points))
}

/// Build one Plotly trace per unique label (sorted), or one
/// trace for all points if no labels. Each trace is a JSON
/// object string ready to drop into a JS array literal.
fn build_traces(points: &[Point3], labels: Option<&[i64]>, n: usize) -> String {
    let groups: Vec<(String, Vec<usize>)> = match labels {
        Some(lbls) => {
            let mut uniq: Vec<i64> = lbls.to_vec();
            uniq.sort_unstable();
            uniq.dedup();
            uniq.into_iter()
                .map(|u| {
                    let idx: Vec<usize> = (0..n).filter(|&i| lbls[i] == u).collect();
                    (format!("class {u}"), idx)
                })
                .collect()
        }
        None => vec![("points".into(), (0..n).collect())],
    };
    let mut out = String::from("[");
    for (i, (name, idx)) in groups.iter().enumerate() {
        if i > 0 {
            out.push(',');
        }
        let color = PALETTE[i % PALETTE.len()];
        let xs = json_num_list(idx.iter().map(|&j| points[j].0));
        let ys = json_num_list(idx.iter().map(|&j| points[j].1));
        let zs = json_num_list(idx.iter().map(|&j| points[j].2));
        out.push_str(r#"{"type":"scatter3d","mode":"markers","name":""#);
        out.push_str(name);
        out.push_str(r#"","x":"#);
        out.push_str(&xs);
        out.push_str(r#","y":"#);
        out.push_str(&ys);
        out.push_str(r#","z":"#);
        out.push_str(&zs);
        out.push_str(r#","marker":{"size":4,"color":""#);
        out.push_str(color);
        out.push_str(r##"","line":{"color":"#1e1e2e","width":1}}}"##);
    }
    out.push(']');
    out
}

fn json_num_list(it: impl IntoIterator<Item = f64>) -> String {
    let mut s = String::from("[");
    for (i, v) in it.into_iter().enumerate() {
        if i > 0 {
            s.push(',');
        }
        if v.is_finite() {
            let _ = write!(s, "{v}");
        } else {
            s.push_str("null");
        }
    }
    s.push(']');
    s
}

/// Per-trace, per-point -> original sample index. The click
/// handler reads `map[curveNumber][pointNumber]` to recover
/// the original sample id when labels split the data into
/// multiple traces.
fn build_cluster_map(labels: Option<&[i64]>, n: usize) -> String {
    let groups: Vec<Vec<usize>> = match labels {
        Some(lbls) => {
            let mut uniq: Vec<i64> = lbls.to_vec();
            uniq.sort_unstable();
            uniq.dedup();
            uniq.into_iter()
                .map(|u| (0..n).filter(|&i| lbls[i] == u).collect())
                .collect()
        }
        None => vec![(0..n).collect()],
    };
    let mut s = String::from("[");
    for (i, g) in groups.iter().enumerate() {
        if i > 0 {
            s.push(',');
        }
        s.push_str(&json_num_list(g.iter().map(|&v| v as f64)));
    }
    s.push(']');
    s
}

/// Hash the data + aux into a stable hex string so multiple
/// plotly3d divs on the same page don't collide. Not
/// cryptographic; only needs uniqueness within a session.
fn stable_uuid(data: &DenseArray, aux: Option<&DenseArray>) -> String {
    use std::hash::{Hash, Hasher};
    let mut h = std::collections::hash_map::DefaultHasher::new();
    for v in data.data() {
        v.to_bits().hash(&mut h);
    }
    if let Some(a) = aux {
        for v in a.data() {
            v.to_bits().hash(&mut h);
        }
    }
    format!("{:016x}", h.finish())
}

/// Wrap the JS payload in a self-contained HTML fragment.
/// The leading marker comment is how the web playground
/// detects a plotly3d payload vs a normal SVG string. Built
/// via push_str to avoid format!'s brace-escaping pain on the
/// embedded JS object literals.
fn emit_html(uuid: &str, traces: &str, map: &str) -> String {
    let mut s = String::with_capacity(4096);
    s.push_str(MARKER);
    s.push_str("\n<div id=\"mlpl-plotly3d-");
    s.push_str(uuid);
    s.push_str("\" style=\"width:100%;height:380px;\"></div>\n<script>(function(){");
    s.push_str("if(typeof Plotly==='undefined'){");
    s.push_str("var d=document.getElementById('mlpl-plotly3d-");
    s.push_str(uuid);
    s.push_str("');");
    s.push_str("if(d)d.innerHTML='<div style=\"padding:1em;color:#f38ba8\">");
    s.push_str("Plotly library not loaded -- include &lt;script src=\"https://cdn.plot.ly/plotly-2.35.2.min.js\"&gt;&lt;/script&gt; in the host page.</div>';");
    s.push_str("return;}");
    s.push_str("var layout={margin:{l:0,r:0,b:0,t:0},paper_bgcolor:'#1e1e2e',plot_bgcolor:'#1e1e2e',font:{color:'#cdd6f4'},");
    s.push_str("scene:{xaxis:{color:'#cdd6f4',gridcolor:'#45475a'},yaxis:{color:'#cdd6f4',gridcolor:'#45475a'},zaxis:{color:'#cdd6f4',gridcolor:'#45475a'}}};");
    s.push_str("var map=");
    s.push_str(map);
    s.push_str(";Plotly.newPlot('mlpl-plotly3d-");
    s.push_str(uuid);
    s.push_str("',");
    s.push_str(traces);
    s.push_str(",layout,{responsive:true,displaylogo:false}).then(function(gd){");
    s.push_str("gd.on('plotly_click',function(evt){");
    s.push_str("if(!evt.points||evt.points.length===0)return;");
    s.push_str("var pt=evt.points[0];");
    s.push_str("var sampleIdx=(map[pt.curveNumber]||[])[pt.pointNumber];");
    s.push_str("if(sampleIdx===undefined)sampleIdx=pt.pointNumber;");
    s.push_str("var label=pt.data.name||'';");
    s.push_str("if(window.mlpl_plotly_click)window.mlpl_plotly_click(sampleIdx,label);");
    s.push_str("else console.log('plotly click: sample #'+sampleIdx+' (label='+label+')');");
    s.push_str("});});})();</script>");
    s
}
