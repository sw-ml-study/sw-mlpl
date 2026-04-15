Phase 2 step 006: Apply and decode for a trained BPE tokenizer.

apply_tokenizer(tok, text): tok is a Value::Tokenizer, text is a Value::Str (or a rank-1 byte array; accept both like train_bpe). Returns a rank-1 DenseArray of token ids. For ByteLevel tokenizers this is exactly tokenize_bytes. For BpeMerges tokenizers, walks the byte sequence applying merges in training order until no more apply (or apply lowest-id merges first -- match standard BPE apply semantics; decide at step start and document).

decode(tok, tokens): inverse. For ByteLevel, identical to decode_bytes. For BpeMerges, recursively expands any merged-id back into its pair until only byte-level ids remain, then recombines into a UTF-8 byte sequence and returns as Value::Str.

Round-trip: for every Value::Str s, decode(tok, apply_tokenizer(tok, s)) == s where tok is any tokenizer trained from the same or a superset corpus. Document the edge case where apply and decode disagree on bytes not seen during training (they still round-trip byte-identically because BPE is lossless at byte level; add a test covering an unseen-byte-sequence input).

The existing apply(model, X) from Saga 11 already exists for models; apply_tokenizer is a separate dispatch to avoid merging tokenizer and model dispatch logic -- keep them as distinct builtins so the eval.rs dispatch stays simple.

TDD:
(1) for a ByteLevel tokenizer, apply_tokenizer matches tokenize_bytes and decode matches decode_bytes on all cases from step 004.
(2) for a small trained BPE tokenizer, round-trip on the training corpus is byte-identical.
(3) round-trip on a byte sequence not in the training corpus still works byte-identically.
(4) token ids never exceed vocab_size after apply.
(5) apply on a Value::Str and the same bytes tokenized returns the same ids.
