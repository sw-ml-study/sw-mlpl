// Regression tests for the 3D inspector derivation -> LaTeX path.
// Guards the class of bug reported on the "User Defined Functions"
// demo step #9: a `def`/`while` line was fed to MathJax as inline
// math and threw "Extra close brace or missing open brace" on its
// literal `{ }`. Run: `node --test` in this dir (or
// `node --test components/web/crates/mlpl-web/js/`).

import { test } from 'node:test';
import assert from 'node:assert/strict';
import { mathRenderable, mlplToLatex, braceSubscripts } from './derivation_latex.js';

// A `\(...\)` payload is MathJax-safe only if its braces balance.
function bracesBalanced(s) {
    let depth = 0;
    for (const ch of s) {
        if (ch === '{') depth += 1;
        else if (ch === '}') {
            depth -= 1;
            if (depth < 0) return false;
        }
    }
    return depth === 0;
}

test('statements / blocks are NOT math-renderable (no MathJax brace error)', () => {
    // The exact step #9 line that broke.
    const fit = 'def u:fit(w0, lr, steps) { w = w0; i = 0; while gt(steps, i) { g = 2 * (w - 3); w = w - lr * g; i = i + 1 }; w }';
    assert.equal(mathRenderable(fit), false, 'def-with-braces must be plain code');
    assert.equal(mathRenderable('def u:fit(w0, lr, steps) { w'), false, 'mis-split def LHS');
    assert.equal(mathRenderable('while gt(steps, i) { g = 2 * (w - 3) }'), false);
    assert.equal(mathRenderable('w = w0; i = 0'), false, 'multi-statement (;)');
    assert.equal(mathRenderable('repeat 5 { x = x + 1 }'), false);
});

test('genuine math expressions ARE renderable', () => {
    assert.equal(mathRenderable('softmax(matmul(Q, transpose(K)) / sqrt(4), 1)'), true);
    assert.equal(mathRenderable('a + b * c'), true);
    assert.equal(mathRenderable('2 * (w - 3)'), true);
});

test('mlplToLatex always yields balanced braces (MathJax-safe)', () => {
    const exprs = [
        'softmax(matmul(Q, transpose(K)) / sqrt(4), 1)',
        'reduce_add(causal_attention(x))',
        'd_model + a_index',
        'sqrt(d)',
        'relu(linear(x))',
    ];
    for (const e of exprs) {
        const out = mlplToLatex(e);
        assert.ok(bracesBalanced(out), `unbalanced braces from "${e}" -> "${out}"`);
    }
});

test('braceSubscripts wraps multi-char subscripts, leaves \\text spans literal', () => {
    assert.equal(braceSubscripts('d_model'), 'd_{model}');
    assert.equal(braceSubscripts('\\text{reduce_add}'), '\\text{reduce_add}');
});
