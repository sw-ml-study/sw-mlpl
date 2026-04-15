Phase 2 step 004: Byte-level tokenizer + Value::Tokenizer scaffolding.

Add tokenize_bytes(s) and decode_bytes(tokens) builtins:
- tokenize_bytes takes a Value::Str and returns a rank-1 DenseArray of byte indices (each 0..=255 as f64).
- decode_bytes takes a rank-1 DenseArray (each cell expected in 0..=255, non-integer or out-of-range cells error) and returns a Value::Str.
- Round-trip: decode_bytes(tokenize_bytes(s)) == s for any str.

Introduce Value::Tokenizer as a new variant on mlpl_eval::Value. Per the milestone doc's open question, decide: dedicated Tokenizer variant, or a generic Value::Handle<T> via TypeId? Pick dedicated Tokenizer for now (fewer unsafe/dyn constructs; Value::Model already established the "dedicated variant per domain" pattern). ModelSpec-like TokenizerSpec enum with a single ByteLevel variant for now; step 005 adds BpeMerges variant.

Introduce a tokenizer() builtin that takes no args and returns Value::Tokenizer(TokenizerSpec::ByteLevel) so users can write `tok = tokenizer()` followed by `apply_tokenizer(tok, "hello")` in step 006. For step 004, apply_tokenizer and decode are not yet implemented; tokenize_bytes/decode_bytes are the direct path.

:describe on a Value::Tokenizer prints the kind (byte-level, or BPE with vocab size in step 005).

TDD:
(1) tokenize_bytes("hello") returns [104, 101, 108, 108, 111].
(2) decode_bytes([104, 101, 108, 108, 111]) returns "hello".
(3) round-trip preserves non-ASCII: decode_bytes(tokenize_bytes("h\xc3\xa9llo")) preserves the UTF-8 bytes.
(4) decode_bytes of an out-of-range value errors clearly.
(5) :describe on Value::Tokenizer(ByteLevel) prints "byte-level tokenizer (vocab=256)".
