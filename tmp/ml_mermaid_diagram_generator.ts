// ml-mermaid-diagram-generator.ts
// Generate Mermaid diagrams for classic ML architectures and training flows.
//
// Usage with Deno:
//   deno run --allow-write ml-mermaid-diagram-generator.ts
//
// Usage with Node + tsx:
//   npm install -D tsx typescript
//   npx tsx ml-mermaid-diagram-generator.ts
//
// Output:
//   ./diagrams/*.mmd

import { mkdirSync, writeFileSync } from "node:fs";
import { join } from "node:path";

type Direction = "TD" | "LR" | "BT" | "RL";
type Shape = "rect" | "round" | "stadium" | "circle" | "diamond" | "subroutine" | "cylinder";
type EdgeStyle = "solid" | "dotted" | "thick";

type Node = {
  id: string;
  label: string;
  shape?: Shape;
  className?: string;
};

type Edge = {
  from: string;
  to: string;
  label?: string;
  style?: EdgeStyle;
};

type Cluster = {
  id: string;
  label: string;
  nodes: Node[];
  edges?: Edge[];
};

type Diagram = {
  name: string;
  title: string;
  direction?: Direction;
  nodes?: Node[];
  edges?: Edge[];
  clusters?: Cluster[];
  classes?: Record<string, string>;
};

const defaultClasses: Record<string, string> = {
  data: "fill:#eef,stroke:#668,stroke-width:1px",
  tensor: "fill:#e8f4ff,stroke:#3973ac,stroke-width:1px",
  op: "fill:#fff7e6,stroke:#b37700,stroke-width:1px",
  block: "fill:#f4ecff,stroke:#7e57c2,stroke-width:1px",
  model: "fill:#ecfff4,stroke:#2e8b57,stroke-width:1px",
  loss: "fill:#ffecec,stroke:#cc4444,stroke-width:2px",
  grad: "fill:#fff0f5,stroke:#cc3377,stroke-width:1px",
  memory: "fill:#f0f0f0,stroke:#666,stroke-width:1px",
  gpu: "fill:#eefaf8,stroke:#088,stroke-width:1px",
  frozen: "fill:#f5f5f5,stroke:#999,stroke-dasharray: 5 5",
  adapter: "fill:#fffbe6,stroke:#aa8800,stroke-width:2px",
};

function escapeLabel(s: string): string {
  return s.replace(/"/g, "&quot;").replace(/\n/g, "<br/>");
}

function renderNode(n: Node): string {
  const label = escapeLabel(n.label);
  switch (n.shape ?? "round") {
    case "rect": return `${n.id}["${label}"]`;
    case "round": return `${n.id}("${label}")`;
    case "stadium": return `${n.id}(["${label}"])`;
    case "circle": return `${n.id}(("${label}"))`;
    case "diamond": return `${n.id}{"${label}"}`;
    case "subroutine": return `${n.id}[["${label}"]]`;
    case "cylinder": return `${n.id}[("${label}")]`;
  }
}

function renderEdge(e: Edge): string {
  const label = e.label ? `|${escapeLabel(e.label)}|` : "";
  switch (e.style ?? "solid") {
    case "solid": return `${e.from} -->${label} ${e.to}`;
    case "dotted": return `${e.from} -.->${label} ${e.to}`;
    case "thick": return `${e.from} ==>${label} ${e.to}`;
  }
}

function renderDiagram(d: Diagram): string {
  const lines: string[] = [];
  lines.push(`---`);
  lines.push(`title: ${d.title}`);
  lines.push(`---`);
  lines.push(`flowchart ${d.direction ?? "LR"}`);
  lines.push("");

  for (const c of d.clusters ?? []) {
    lines.push(`  subgraph ${c.id}["${escapeLabel(c.label)}"]`);
    for (const n of c.nodes) lines.push(`    ${renderNode(n)}`);
    for (const e of c.edges ?? []) lines.push(`    ${renderEdge(e)}`);
    lines.push(`  end`);
    lines.push("");
  }

  for (const n of d.nodes ?? []) lines.push(`  ${renderNode(n)}`);
  if ((d.nodes ?? []).length) lines.push("");

  for (const e of d.edges ?? []) lines.push(`  ${renderEdge(e)}`);
  lines.push("");

  const mergedClasses = { ...defaultClasses, ...(d.classes ?? {}) };
  for (const [name, style] of Object.entries(mergedClasses)) {
    lines.push(`  classDef ${name} ${style};`);
  }

  const allNodes = [
    ...(d.nodes ?? []),
    ...(d.clusters ?? []).flatMap(c => c.nodes),
  ];
  for (const n of allNodes) {
    if (n.className) lines.push(`  class ${n.id} ${n.className};`);
  }

  return lines.join("\n") + "\n";
}

function seqEdges(ids: string[], label?: string): Edge[] {
  return ids.slice(0, -1).map((id, i) => ({ from: id, to: ids[i + 1], label }));
}

const diagrams: Diagram[] = [
  {
    name: "01_linear_regression",
    title: "Linear Regression Training Loop",
    direction: "LR",
    nodes: [
      { id: "x", label: "Input features\nx", className: "data" },
      { id: "w", label: "Weights\nw", className: "tensor" },
      { id: "pred", label: "Prediction\ny = wx + b", className: "op" },
      { id: "loss", label: "Loss\nMSE", shape: "diamond", className: "loss" },
      { id: "grad", label: "Gradient", className: "grad" },
      { id: "opt", label: "Optimizer\nupdate w,b", className: "op" },
    ],
    edges: [
      { from: "x", to: "pred" },
      { from: "w", to: "pred" },
      { from: "pred", to: "loss" },
      { from: "loss", to: "grad", style: "dotted" },
      { from: "grad", to: "opt", style: "dotted" },
      { from: "opt", to: "w", style: "dotted", label: "learn" },
    ],
  },
  {
    name: "02_logistic_regression",
    title: "Logistic Regression",
    direction: "LR",
    nodes: [
      { id: "x", label: "Features", className: "data" },
      { id: "linear", label: "Linear score\nz = wx + b", className: "op" },
      { id: "sigmoid", label: "Sigmoid\nprobability", className: "op" },
      { id: "threshold", label: "Threshold", className: "op" },
      { id: "class_label", label: "Class label", className: "data" },
    ],
    edges: seqEdges(["x", "linear", "sigmoid", "threshold", "class_label"]),
  },
  {
    name: "03_decision_tree",
    title: "Decision Tree",
    direction: "TD",
    nodes: [
      { id: "root", label: "Feature A < t?", shape: "diamond", className: "op" },
      { id: "left", label: "Feature B < t?", shape: "diamond", className: "op" },
      { id: "right", label: "Class 2", shape: "stadium", className: "data" },
      { id: "l1", label: "Class 0", shape: "stadium", className: "data" },
      { id: "l2", label: "Class 1", shape: "stadium", className: "data" },
    ],
    edges: [
      { from: "root", to: "left", label: "yes" },
      { from: "root", to: "right", label: "no" },
      { from: "left", to: "l1", label: "yes" },
      { from: "left", to: "l2", label: "no" },
    ],
  },
  {
    name: "04_random_forest",
    title: "Random Forest",
    direction: "LR",
    nodes: [
      { id: "batch", label: "Input row", className: "data" },
      { id: "t1", label: "Tree 1", className: "model" },
      { id: "t2", label: "Tree 2", className: "model" },
      { id: "t3", label: "Tree N", className: "model" },
      { id: "vote", label: "Vote / average", className: "op" },
      { id: "out", label: "Prediction", className: "data" },
    ],
    edges: [
      { from: "batch", to: "t1" }, { from: "batch", to: "t2" }, { from: "batch", to: "t3" },
      { from: "t1", to: "vote" }, { from: "t2", to: "vote" }, { from: "t3", to: "vote" },
      { from: "vote", to: "out" },
    ],
  },
  {
    name: "05_svm",
    title: "Support Vector Machine",
    direction: "LR",
    nodes: [
      { id: "points", label: "Labeled points", className: "data" },
      { id: "kernel", label: "Kernel / feature map", className: "op" },
      { id: "margin", label: "Max-margin separator", className: "model" },
      { id: "sv", label: "Support vectors", className: "tensor" },
      { id: "pred", label: "Class decision", className: "data" },
    ],
    edges: seqEdges(["points", "kernel", "margin", "pred"]).concat([{ from: "sv", to: "margin", label: "defines" }]),
  },
  {
    name: "06_perceptron",
    title: "Perceptron",
    direction: "LR",
    nodes: [
      { id: "x1", label: "x1", className: "data" },
      { id: "x2", label: "x2", className: "data" },
      { id: "x3", label: "x3", className: "data" },
      { id: "sum", label: "Weighted sum", shape: "circle", className: "op" },
      { id: "act", label: "Activation", className: "op" },
      { id: "y", label: "Output", className: "data" },
    ],
    edges: [
      { from: "x1", to: "sum", label: "w1" },
      { from: "x2", to: "sum", label: "w2" },
      { from: "x3", to: "sum", label: "w3" },
      { from: "sum", to: "act" },
      { from: "act", to: "y" },
    ],
  },
  {
    name: "07_mlp",
    title: "Multilayer Perceptron",
    direction: "LR",
    nodes: [
      { id: "input", label: "Input\n[B,F]", className: "tensor" },
      { id: "h1", label: "Dense + activation", className: "block" },
      { id: "h2", label: "Dense + activation", className: "block" },
      { id: "head", label: "Output head", className: "block" },
      { id: "out", label: "Prediction", className: "data" },
    ],
    edges: seqEdges(["input", "h1", "h2", "head", "out"]),
  },
  {
    name: "08_cnn",
    title: "Convolutional Neural Network",
    direction: "LR",
    nodes: [
      { id: "img", label: "Image\n[H,W,C]", className: "tensor" },
      { id: "conv1", label: "Conv kernels", className: "block" },
      { id: "maps", label: "Feature maps", className: "tensor" },
      { id: "pool", label: "Pooling", className: "op" },
      { id: "conv2", label: "Deeper convs", className: "block" },
      { id: "flat", label: "Flatten / GAP", className: "op" },
      { id: "cls", label: "Classifier", className: "model" },
    ],
    edges: seqEdges(["img", "conv1", "maps", "pool", "conv2", "flat", "cls"]),
  },
  {
    name: "09_resnet",
    title: "Residual Network Block",
    direction: "LR",
    nodes: [
      { id: "x", label: "x", className: "tensor" },
      { id: "conv1", label: "Conv + norm + act", className: "block" },
      { id: "conv2", label: "Conv + norm", className: "block" },
      { id: "add", label: "Add", shape: "circle", className: "op" },
      { id: "act", label: "Activation", className: "op" },
      { id: "y", label: "y", className: "tensor" },
    ],
    edges: [
      { from: "x", to: "conv1" }, { from: "conv1", to: "conv2" }, { from: "conv2", to: "add" },
      { from: "x", to: "add", label: "skip", style: "thick" },
      { from: "add", to: "act" }, { from: "act", to: "y" },
    ],
  },
  {
    name: "10_rnn",
    title: "Recurrent Neural Network Unrolled Through Time",
    direction: "LR",
    nodes: [
      { id: "x1", label: "x₁", className: "data" }, { id: "x2", label: "x₂", className: "data" }, { id: "x3", label: "x₃", className: "data" },
      { id: "h1", label: "RNN cell\nh₁", className: "block" }, { id: "h2", label: "RNN cell\nh₂", className: "block" }, { id: "h3", label: "RNN cell\nh₃", className: "block" },
      { id: "y1", label: "y₁", className: "data" }, { id: "y2", label: "y₂", className: "data" }, { id: "y3", label: "y₃", className: "data" },
    ],
    edges: [
      { from: "x1", to: "h1" }, { from: "x2", to: "h2" }, { from: "x3", to: "h3" },
      { from: "h1", to: "h2", label: "state" }, { from: "h2", to: "h3", label: "state" },
      { from: "h1", to: "y1" }, { from: "h2", to: "y2" }, { from: "h3", to: "y3" },
    ],
  },
  {
    name: "11_lstm",
    title: "LSTM Cell",
    direction: "LR",
    nodes: [
      { id: "x", label: "xₜ", className: "data" },
      { id: "hprev", label: "hₜ₋₁", className: "tensor" },
      { id: "cprev", label: "cₜ₋₁", className: "tensor" },
      { id: "fg", label: "Forget gate", className: "op" },
      { id: "ig", label: "Input gate", className: "op" },
      { id: "cand", label: "Candidate", className: "op" },
      { id: "cell", label: "Cell state\ncₜ", className: "memory" },
      { id: "og", label: "Output gate", className: "op" },
      { id: "h", label: "hₜ", className: "tensor" },
    ],
    edges: [
      { from: "x", to: "fg" }, { from: "hprev", to: "fg" },
      { from: "x", to: "ig" }, { from: "hprev", to: "ig" },
      { from: "x", to: "cand" }, { from: "hprev", to: "cand" },
      { from: "cprev", to: "cell", label: "forget / keep", style: "thick" },
      { from: "fg", to: "cell" }, { from: "ig", to: "cell" }, { from: "cand", to: "cell" },
      { from: "cell", to: "og" }, { from: "og", to: "h" },
    ],
  },
  {
    name: "12_attention",
    title: "Scaled Dot-Product Attention",
    direction: "LR",
    nodes: [
      { id: "q", label: "Query Q", className: "tensor" },
      { id: "k", label: "Key K", className: "tensor" },
      { id: "v", label: "Value V", className: "tensor" },
      { id: "score", label: "QKᵀ / √d", className: "op" },
      { id: "softmax", label: "Softmax\nattention weights", className: "op" },
      { id: "weighted", label: "Weighted sum\n× V", className: "op" },
      { id: "out", label: "Context vectors", className: "tensor" },
    ],
    edges: [
      { from: "q", to: "score" }, { from: "k", to: "score" },
      { from: "score", to: "softmax" }, { from: "softmax", to: "weighted" },
      { from: "v", to: "weighted" }, { from: "weighted", to: "out" },
    ],
  },
  {
    name: "13_multi_head_attention",
    title: "Multi-Head Attention",
    direction: "LR",
    nodes: [
      { id: "x", label: "Token embeddings\n[B,T,D]", className: "tensor" },
      { id: "h1", label: "Head 1", className: "block" },
      { id: "h2", label: "Head 2", className: "block" },
      { id: "hn", label: "Head N", className: "block" },
      { id: "cat", label: "Concat", className: "op" },
      { id: "proj", label: "Output projection", className: "op" },
      { id: "y", label: "Mixed representation", className: "tensor" },
    ],
    edges: [
      { from: "x", to: "h1" }, { from: "x", to: "h2" }, { from: "x", to: "hn" },
      { from: "h1", to: "cat" }, { from: "h2", to: "cat" }, { from: "hn", to: "cat" },
      { from: "cat", to: "proj" }, { from: "proj", to: "y" },
    ],
  },
  {
    name: "14_transformer_encoder",
    title: "Transformer Encoder Stack",
    direction: "TD",
    nodes: [
      { id: "tokens", label: "Tokens", className: "data" },
      { id: "embed", label: "Token + position embeddings", className: "tensor" },
      { id: "attn", label: "Self-attention", className: "block" },
      { id: "addnorm1", label: "Add + Norm", className: "op" },
      { id: "ffn", label: "Feed-forward MLP", className: "block" },
      { id: "addnorm2", label: "Add + Norm", className: "op" },
      { id: "repeat", label: "Repeat N layers", shape: "subroutine", className: "model" },
      { id: "encoded", label: "Encoded sequence", className: "tensor" },
    ],
    edges: seqEdges(["tokens", "embed", "attn", "addnorm1", "ffn", "addnorm2", "repeat", "encoded"]),
  },
  {
    name: "15_transformer_decoder",
    title: "Transformer Decoder Block",
    direction: "TD",
    nodes: [
      { id: "targets", label: "Previous tokens", className: "data" },
      { id: "masked", label: "Masked self-attention", className: "block" },
      { id: "cross", label: "Cross-attention", className: "block" },
      { id: "encoder", label: "Encoder memory", className: "memory" },
      { id: "ffn", label: "Feed-forward MLP", className: "block" },
      { id: "logits", label: "Next-token logits", className: "tensor" },
    ],
    edges: [
      { from: "targets", to: "masked" }, { from: "masked", to: "cross" },
      { from: "encoder", to: "cross", label: "K,V" },
      { from: "cross", to: "ffn" }, { from: "ffn", to: "logits" },
    ],
  },
  {
    name: "16_encoder_decoder_transformer",
    title: "Encoder-Decoder Transformer",
    direction: "LR",
    clusters: [
      { id: "enc", label: "Encoder", nodes: [
        { id: "src", label: "Source tokens", className: "data" },
        { id: "encstack", label: "N encoder blocks", className: "model" },
        { id: "memory", label: "Encoder memory", className: "memory" },
      ], edges: seqEdges(["src", "encstack", "memory"]) },
      { id: "dec", label: "Decoder", nodes: [
        { id: "tgt", label: "Target prefix", className: "data" },
        { id: "decstack", label: "N decoder blocks", className: "model" },
        { id: "out", label: "Next token", className: "data" },
      ], edges: seqEdges(["tgt", "decstack", "out"]) },
    ],
    edges: [{ from: "memory", to: "decstack", label: "cross-attend" }],
  },
  {
    name: "17_gpt_decoder_only",
    title: "GPT-Style Decoder-Only Transformer",
    direction: "LR",
    nodes: [
      { id: "prompt", label: "Prompt tokens", className: "data" },
      { id: "embed", label: "Token + position embeddings", className: "tensor" },
      { id: "blocks", label: "Repeated causal decoder blocks", className: "model" },
      { id: "lmhead", label: "LM head", className: "op" },
      { id: "logits", label: "Next-token logits", className: "tensor" },
      { id: "sample", label: "Sample / argmax", className: "op" },
      { id: "next", label: "Next token", className: "data" },
      { id: "kv", label: "KV cache", shape: "cylinder", className: "memory" },
    ],
    edges: [
      ...seqEdges(["prompt", "embed", "blocks", "lmhead", "logits", "sample", "next"]),
      { from: "blocks", to: "kv", label: "write", style: "dotted" },
      { from: "kv", to: "blocks", label: "read", style: "dotted" },
      { from: "next", to: "prompt", label: "append", style: "dotted" },
    ],
  },
  {
    name: "18_moe",
    title: "Mixture of Experts Layer",
    direction: "LR",
    nodes: [
      { id: "x", label: "Token representation", className: "tensor" },
      { id: "router", label: "Router", className: "op" },
      { id: "e1", label: "Expert 1", className: "model" },
      { id: "e2", label: "Expert 2", className: "model" },
      { id: "en", label: "Expert N", className: "frozen" },
      { id: "combine", label: "Weighted combine", className: "op" },
      { id: "y", label: "Output", className: "tensor" },
    ],
    edges: [
      { from: "x", to: "router" }, { from: "router", to: "e1", label: "top-k" }, { from: "router", to: "e2", label: "top-k" }, { from: "router", to: "en", style: "dotted", label: "usually skipped" },
      { from: "e1", to: "combine" }, { from: "e2", to: "combine" }, { from: "combine", to: "y" },
    ],
  },
  {
    name: "19_rag",
    title: "Retrieval-Augmented Generation",
    direction: "LR",
    nodes: [
      { id: "q", label: "User query", className: "data" },
      { id: "embed", label: "Embed query", className: "op" },
      { id: "vdb", label: "Vector DB", shape: "cylinder", className: "memory" },
      { id: "docs", label: "Retrieved chunks", className: "data" },
      { id: "prompt", label: "Augmented prompt", className: "data" },
      { id: "llm", label: "LLM", className: "model" },
      { id: "answer", label: "Answer + citations", className: "data" },
    ],
    edges: seqEdges(["q", "embed", "vdb", "docs", "prompt", "llm", "answer"]),
  },
  {
    name: "20_agent_loop",
    title: "Agentic LLM Loop",
    direction: "LR",
    nodes: [
      { id: "goal", label: "Goal", className: "data" },
      { id: "planner", label: "Planner", className: "model" },
      { id: "tool", label: "Tool call", className: "op" },
      { id: "obs", label: "Observation", className: "data" },
      { id: "memory", label: "Memory / scratchpad", shape: "cylinder", className: "memory" },
      { id: "answer", label: "Final response", className: "data" },
    ],
    edges: [
      { from: "goal", to: "planner" }, { from: "planner", to: "tool" }, { from: "tool", to: "obs" },
      { from: "obs", to: "memory" }, { from: "memory", to: "planner", style: "dotted" },
      { from: "planner", to: "answer", label: "done" },
    ],
  },
  {
    name: "21_vit",
    title: "Vision Transformer",
    direction: "LR",
    nodes: [
      { id: "img", label: "Image", className: "data" },
      { id: "patch", label: "Split into patches", className: "op" },
      { id: "embed", label: "Patch embeddings", className: "tensor" },
      { id: "cls", label: "[CLS] token", className: "tensor" },
      { id: "enc", label: "Transformer encoder", className: "model" },
      { id: "head", label: "Classification head", className: "op" },
    ],
    edges: seqEdges(["img", "patch", "embed", "enc", "head"]).concat([{ from: "cls", to: "enc" }]),
  },
  {
    name: "22_unet",
    title: "U-Net",
    direction: "LR",
    nodes: [
      { id: "input", label: "Input image / latent", className: "tensor" },
      { id: "down1", label: "Down block 1", className: "block" },
      { id: "down2", label: "Down block 2", className: "block" },
      { id: "bottleneck", label: "Bottleneck", className: "model" },
      { id: "up2", label: "Up block 2", className: "block" },
      { id: "up1", label: "Up block 1", className: "block" },
      { id: "out", label: "Output", className: "tensor" },
    ],
    edges: [
      ...seqEdges(["input", "down1", "down2", "bottleneck", "up2", "up1", "out"]),
      { from: "down2", to: "up2", label: "skip", style: "thick" },
      { from: "down1", to: "up1", label: "skip", style: "thick" },
    ],
  },
  {
    name: "23_diffusion",
    title: "Diffusion Model Training and Sampling",
    direction: "LR",
    nodes: [
      { id: "clean", label: "Clean sample x₀", className: "data" },
      { id: "noise", label: "Add noise\nforward process", className: "op" },
      { id: "xt", label: "Noisy sample xₜ", className: "tensor" },
      { id: "unet", label: "Denoiser U-Net", className: "model" },
      { id: "pred", label: "Predicted noise", className: "tensor" },
      { id: "loss", label: "Noise prediction loss", shape: "diamond", className: "loss" },
      { id: "sample", label: "Reverse denoising loop", className: "op" },
    ],
    edges: [
      ...seqEdges(["clean", "noise", "xt", "unet", "pred", "loss"]),
      { from: "unet", to: "sample", label: "inference" },
      { from: "sample", to: "clean", label: "generate", style: "dotted" },
    ],
  },
  {
    name: "24_clip",
    title: "CLIP Dual Encoder",
    direction: "LR",
    nodes: [
      { id: "image", label: "Image", className: "data" },
      { id: "text", label: "Text", className: "data" },
      { id: "ienc", label: "Image encoder", className: "model" },
      { id: "tenc", label: "Text encoder", className: "model" },
      { id: "space", label: "Shared embedding space", className: "tensor" },
      { id: "contrast", label: "Contrastive loss", shape: "diamond", className: "loss" },
    ],
    edges: [
      { from: "image", to: "ienc" }, { from: "text", to: "tenc" },
      { from: "ienc", to: "space" }, { from: "tenc", to: "space" },
      { from: "space", to: "contrast" },
    ],
  },
  {
    name: "25_vlm",
    title: "Vision-Language Model",
    direction: "LR",
    nodes: [
      { id: "img", label: "Image", className: "data" },
      { id: "vision", label: "Vision encoder", className: "model" },
      { id: "proj", label: "Projector / adapter", className: "adapter" },
      { id: "itokens", label: "Image tokens", className: "tensor" },
      { id: "text", label: "Text tokens", className: "data" },
      { id: "llm", label: "LLM backbone", className: "model" },
      { id: "answer", label: "Multimodal answer", className: "data" },
    ],
    edges: [
      ...seqEdges(["img", "vision", "proj", "itokens", "llm", "answer"]),
      { from: "text", to: "llm" },
    ],
  },
  {
    name: "26_mamba_ssm",
    title: "State Space / Mamba-Style Sequence Model",
    direction: "LR",
    nodes: [
      { id: "tokens", label: "Token stream", className: "data" },
      { id: "proj", label: "Input projection", className: "op" },
      { id: "select", label: "Selective parameters", className: "op" },
      { id: "scan", label: "Selective scan\nstate update", className: "block" },
      { id: "state", label: "Compressed state", shape: "cylinder", className: "memory" },
      { id: "out", label: "Output sequence", className: "tensor" },
    ],
    edges: [
      ...seqEdges(["tokens", "proj", "select", "scan", "out"]),
      { from: "scan", to: "state", label: "write", style: "dotted" },
      { from: "state", to: "scan", label: "read", style: "dotted" },
    ],
  },
  {
    name: "27_training_loop",
    title: "Generic Neural Network Training Loop",
    direction: "LR",
    nodes: [
      { id: "data", label: "Dataset", shape: "cylinder", className: "data" },
      { id: "batch", label: "Mini-batch", className: "tensor" },
      { id: "model", label: "Model forward pass", className: "model" },
      { id: "loss", label: "Loss", shape: "diamond", className: "loss" },
      { id: "back", label: "Backprop", className: "grad" },
      { id: "opt", label: "Optimizer step", className: "op" },
      { id: "weights", label: "Weights", shape: "cylinder", className: "memory" },
    ],
    edges: [
      ...seqEdges(["data", "batch", "model", "loss", "back", "opt", "weights"]),
      { from: "weights", to: "model", label: "updated params", style: "dotted" },
    ],
  },
  {
    name: "28_backprop",
    title: "Forward Pass and Backpropagation",
    direction: "LR",
    nodes: [
      { id: "x", label: "Input", className: "data" },
      { id: "l1", label: "Layer 1", className: "block" },
      { id: "l2", label: "Layer 2", className: "block" },
      { id: "l3", label: "Layer 3", className: "block" },
      { id: "loss", label: "Loss", shape: "diamond", className: "loss" },
    ],
    edges: [
      ...seqEdges(["x", "l1", "l2", "l3", "loss"], "forward"),
      { from: "loss", to: "l3", label: "grad", style: "dotted" },
      { from: "l3", to: "l2", label: "grad", style: "dotted" },
      { from: "l2", to: "l1", label: "grad", style: "dotted" },
    ],
  },
  {
    name: "29_data_parallel_training",
    title: "Data Parallel Training",
    direction: "LR",
    nodes: [
      { id: "batch", label: "Large batch", className: "tensor" },
      { id: "split", label: "Split batch", className: "op" },
      { id: "gpu1", label: "GPU 1\nmodel replica", className: "gpu" },
      { id: "gpu2", label: "GPU 2\nmodel replica", className: "gpu" },
      { id: "gpun", label: "GPU N\nmodel replica", className: "gpu" },
      { id: "allreduce", label: "All-reduce gradients", className: "op" },
      { id: "update", label: "Synchronized update", className: "op" },
    ],
    edges: [
      { from: "batch", to: "split" },
      { from: "split", to: "gpu1" }, { from: "split", to: "gpu2" }, { from: "split", to: "gpun" },
      { from: "gpu1", to: "allreduce" }, { from: "gpu2", to: "allreduce" }, { from: "gpun", to: "allreduce" },
      { from: "allreduce", to: "update" },
    ],
  },
  {
    name: "30_tensor_parallel_training",
    title: "Tensor Parallel Training",
    direction: "LR",
    nodes: [
      { id: "x", label: "Input activation", className: "tensor" },
      { id: "shard1", label: "Weight shard A\nGPU 1", className: "gpu" },
      { id: "shard2", label: "Weight shard B\nGPU 2", className: "gpu" },
      { id: "partial1", label: "Partial result A", className: "tensor" },
      { id: "partial2", label: "Partial result B", className: "tensor" },
      { id: "merge", label: "Gather / reduce", className: "op" },
      { id: "y", label: "Output activation", className: "tensor" },
    ],
    edges: [
      { from: "x", to: "shard1" }, { from: "x", to: "shard2" },
      { from: "shard1", to: "partial1" }, { from: "shard2", to: "partial2" },
      { from: "partial1", to: "merge" }, { from: "partial2", to: "merge" },
      { from: "merge", to: "y" },
    ],
  },
  {
    name: "31_pipeline_parallel_training",
    title: "Pipeline Parallel Training",
    direction: "LR",
    nodes: [
      { id: "mb1", label: "Microbatch 1", className: "tensor" },
      { id: "s1", label: "Stage 1\nGPU 1", className: "gpu" },
      { id: "s2", label: "Stage 2\nGPU 2", className: "gpu" },
      { id: "s3", label: "Stage 3\nGPU 3", className: "gpu" },
      { id: "loss", label: "Loss", shape: "diamond", className: "loss" },
      { id: "bubble", label: "Pipeline bubbles / stalls", className: "op" },
    ],
    edges: [
      ...seqEdges(["mb1", "s1", "s2", "s3", "loss"]),
      { from: "bubble", to: "s1", style: "dotted" },
      { from: "bubble", to: "s2", style: "dotted" },
      { from: "bubble", to: "s3", style: "dotted" },
    ],
  },
  {
    name: "32_lora",
    title: "LoRA Fine-Tuning",
    direction: "LR",
    nodes: [
      { id: "x", label: "Input", className: "tensor" },
      { id: "base", label: "Frozen base weight W", className: "frozen" },
      { id: "a", label: "Low-rank A", className: "adapter" },
      { id: "b", label: "Low-rank B", className: "adapter" },
      { id: "sum", label: "W + BA", shape: "circle", className: "op" },
      { id: "y", label: "Output", className: "tensor" },
    ],
    edges: [
      { from: "x", to: "base" }, { from: "base", to: "sum" },
      { from: "x", to: "a" }, { from: "a", to: "b" }, { from: "b", to: "sum" },
      { from: "sum", to: "y" },
    ],
  },
  {
    name: "33_qlora",
    title: "QLoRA Fine-Tuning",
    direction: "LR",
    nodes: [
      { id: "x", label: "Input", className: "tensor" },
      { id: "qbase", label: "Quantized frozen base\n4-bit", className: "frozen" },
      { id: "deq", label: "Dequantize for compute", className: "op" },
      { id: "lora", label: "Trainable LoRA adapters", className: "adapter" },
      { id: "out", label: "Output", className: "tensor" },
      { id: "mem", label: "Lower VRAM use", shape: "cylinder", className: "memory" },
    ],
    edges: [
      { from: "x", to: "qbase" }, { from: "qbase", to: "deq" }, { from: "deq", to: "out" },
      { from: "x", to: "lora" }, { from: "lora", to: "out" },
      { from: "qbase", to: "mem", style: "dotted" },
    ],
  },
  {
    name: "34_rlhf",
    title: "RLHF Pipeline",
    direction: "LR",
    nodes: [
      { id: "pre", label: "Pretrained model", className: "model" },
      { id: "sft", label: "Supervised fine-tuning", className: "op" },
      { id: "prefs", label: "Human preferences", className: "data" },
      { id: "reward", label: "Reward model", className: "model" },
      { id: "rl", label: "PPO / RL update", className: "op" },
      { id: "aligned", label: "Aligned assistant", className: "model" },
    ],
    edges: [
      ...seqEdges(["pre", "sft", "rl", "aligned"]),
      { from: "prefs", to: "reward" },
      { from: "reward", to: "rl", label: "reward signal" },
    ],
  },
  {
    name: "35_dpo",
    title: "Direct Preference Optimization",
    direction: "LR",
    nodes: [
      { id: "prompt", label: "Prompt", className: "data" },
      { id: "chosen", label: "Chosen answer", className: "data" },
      { id: "rejected", label: "Rejected answer", className: "data" },
      { id: "policy", label: "Policy model", className: "model" },
      { id: "ref", label: "Reference model", className: "frozen" },
      { id: "loss", label: "Preference loss", shape: "diamond", className: "loss" },
    ],
    edges: [
      { from: "prompt", to: "policy" }, { from: "chosen", to: "loss" }, { from: "rejected", to: "loss" },
      { from: "policy", to: "loss" }, { from: "ref", to: "loss" },
    ],
  },
  {
    name: "36_self_play_training",
    title: "Self-Play / Synthetic Data Training Loop",
    direction: "LR",
    nodes: [
      { id: "model", label: "Current model", className: "model" },
      { id: "task", label: "Task generator", className: "op" },
      { id: "attempt", label: "Attempt / solution", className: "data" },
      { id: "judge", label: "Judge / verifier", className: "model" },
      { id: "buffer", label: "Replay buffer", shape: "cylinder", className: "memory" },
      { id: "train", label: "Train on selected examples", className: "op" },
    ],
    edges: [
      { from: "model", to: "task" }, { from: "task", to: "attempt" }, { from: "attempt", to: "judge" },
      { from: "judge", to: "buffer", label: "score/filter" }, { from: "buffer", to: "train" },
      { from: "train", to: "model", label: "improve", style: "dotted" },
    ],
  },
  {
    name: "37_grokking",
    title: "Grokking Training Dynamics",
    direction: "LR",
    nodes: [
      { id: "train", label: "Training accuracy rises", className: "data" },
      { id: "memorize", label: "Memorization plateau", className: "op" },
      { id: "phase", label: "Phase transition", className: "diamond" },
      { id: "generalize", label: "Validation accuracy jumps", className: "data" },
      { id: "simple", label: "Simpler internal rule", className: "model" },
    ],
    edges: seqEdges(["train", "memorize", "phase", "generalize", "simple"]),
  },
  {
    name: "38_superposition",
    title: "Feature Superposition",
    direction: "LR",
    nodes: [
      { id: "features", label: "Many sparse features", className: "data" },
      { id: "basis", label: "Limited neuron basis", className: "tensor" },
      { id: "mix", label: "Features share directions", className: "op" },
      { id: "activation", label: "Activation pattern", className: "tensor" },
      { id: "interp", label: "Interpretability challenge", className: "loss" },
    ],
    edges: seqEdges(["features", "basis", "mix", "activation", "interp"]),
  },
];

function writeAll(outputDir = "diagrams") {
  mkdirSync(outputDir, { recursive: true });
  for (const diagram of diagrams) {
    const path = join(outputDir, `${diagram.name}.mmd`);
    writeFileSync(path, renderDiagram(diagram), "utf8");
    console.log(`wrote ${path}`);
  }

  const index = diagrams
    .map(d => `- [${d.title}](./${d.name}.mmd)`)
    .join("\n");
  writeFileSync(join(outputDir, "README.md"), `# ML Mermaid Diagrams\n\n${index}\n`, "utf8");
}

writeAll();
