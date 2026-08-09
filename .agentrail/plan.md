# Saga: codec-mlpb-integrity

../demo-algorithms adopted the MLPB v1 typed-native codec and
asked (secondary) for a "stronger MLPB integrity/checksum" beyond
the current magic + version + payload_len header. Add a CRC32
trailer over the payload as MLPB v2, backward-compatible on read.

Format:
- v1 (existing): `MLPB`(4) + version=1 + payload_len(u32 LE) + payload
- v2 (new):      `MLPB`(4) + version=2 + payload_len(u32 LE) + payload + crc32(u32 LE over payload)

`to_native` emits v2 (integrity by default). `parse_native` /
decode accepts BOTH: v1 decodes with no checksum (already-adopted
data stays readable); v2 verifies the CRC32 and errors on
mismatch. All failures stay err Results (never panic).

Isolate the concern in a new `native_integrity.rs`
(crc32 + read_header + verify_checksum) so `to_native`/`decode`
stay small and every native_* module keeps <=4 fns. crc32 is
bitwise IEEE (poly 0xEDB88320), no table, no new dependency.

## Steps
1. mlpb-checksum -- add native_integrity.rs; bump VERSION to 2;
   to_native appends the CRC32 trailer; decode reads the header via
   read_header (accepting v1/v2) and calls verify_checksum. Update
   the version-byte assertion to v2. TDD: v2 round-trips; a
   corrupted payload byte -> err (checksum mismatch); a synthetic
   v1 buffer still decodes; a truncated/missing checksum -> err.
   Update encode/decode module docs.
2. mlpb-close -- docs (companion-demo-algorithms if present,
   Capability Matrix row, wiki errata, q-and-a), report that
   demo-algorithms can adopt MLPB v2 integrity; --done.
