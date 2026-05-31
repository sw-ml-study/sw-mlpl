// Pure MLPL-expression -> LaTeX helpers for the 3D inspector's
// derivation view, extracted from stage3d.js so they can be unit-
// tested under node (see derivation_latex.test.js). This guards the
// class of bug where a demo line (e.g. a `def`/`while` block) is fed
// to MathJax as inline math and throws "Extra close brace or missing
// open brace" on the literal `{ }`.
//
// ES module: imported by stage3d.js in the browser (<script
// type="module">) and by the node test (js/package.json sets
// "type":"module"). No THREE / DOM deps so it loads cleanly in node.

// True when `s` is a math expression MathJax can typeset. MLPL
// statements / blocks (def, while, if, for, repeat, brace blocks, or
// multi-statement `;` lines) are NOT -- their literal `{ }` / `;`
// make MathJax throw, so callers render them as plain code instead.
export function mathRenderable(s) {
    return !/[{};]/.test(s) && !/^\s*(def|while|if|for|repeat)\b/.test(s);
}

// MathJax treats only the first char after `_` as the subscript, so
// `d_model` renders as d-sub-m followed by full-size "odel". Wrap
// multi-char (and single-char, harmlessly) subscripts in braces:
// `d_model` -> `d_{model}`. The leading alternation consumes whole
// `\text{...}` spans untouched so underscores in function names
// (reduce_add, causal_attention) stay literal text.
export function braceSubscripts(s) {
    return s.replace(/\\text\{[^}]*\}|_([A-Za-z][A-Za-z0-9]*)/g,
        (m, sub) => (sub ? `_{${sub}}` : m));
}

// Convert an MLPL RHS expression to LaTeX. Iterative innermost-first
// replacement so nested calls collapse one layer per pass:
//   softmax(matmul(Q, transpose(K)) / sqrt(4), 1)
//     -> \text{softmax}(Q \cdot K^T / \sqrt{4})
// Function calls without a dedicated rewrite get \text{name}(args).
export function mlplToLatex(expr) {
    if (!expr) return '';
    let cur = expr;
    let prev = '';
    let guard = 0;
    while (cur !== prev && guard < 16) {
        prev = cur;
        cur = cur.replace(/transpose\(([^()]+)\)/g, '$1^T');
        cur = cur.replace(/sqrt\(([^()]+)\)/g, '\\sqrt{$1}');
        cur = cur.replace(/matmul\(([^,()]+),\s*([^()]+)\)/g, '$1 \\cdot $2');
        cur = cur.replace(/softmax\(([^()]+?),\s*\d+\)/g, '\\text{softmax}($1)');
        cur = cur.replace(/(?<![\\a-zA-Z_])([a-z_]\w*)\(([^()]*)\)/g, '\\text{$1}($2)');
        guard += 1;
    }
    return braceSubscripts(cur);
}
