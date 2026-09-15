#pragma once

#include <cstdint>
#include <nlohmann/json.hpp>
#include <string>
#include <vector>

namespace quantum {
namespace backends {
namespace safe_pickle {

// A minimal, deliberately incomplete Python pickle codec, covering only
// what QMIO's wire protocol needs: encoding a 2-tuple of strings as the
// request payload, and decoding a reply built from primitive types
// (None, bool, int, float, str, bytes, list, dict, tuple).
//
// SAFETY: this only ever builds data. Any opcode that can construct an
// arbitrary object or execute code -- REDUCE, GLOBAL/STACK_GLOBAL, BUILD,
// INST, OBJ, NEWOBJ[_EX], EXT1/2/4, PERSID/BINPERSID -- is explicitly
// rejected with an exception rather than interpreted, the same safety
// property Python's own pickle.loads() does NOT have. This mirrors the
// design of Rust's serde-pickle crate, which the reference QMIO client
// (polypus, github.com/Bahia-Software/polypus) is built on specifically
// for this reason. Do not "complete" this decoder by adding those
// opcodes back.
//
// Known limitations (acceptable for a counts-reply payload, worth
// knowing if you extend this to something more general):
//   - Memoization (PUT/BINPUT/GET/BINGET/MEMOIZE) is supported, but a
//     memoized object is stored as a COPY at PUT-time, not a live
//     reference. A pickle that mutates an object *after* memoizing it
//     and expects a later GET to see that mutation will not round-trip
//     correctly. Real picklers building a plain dict/list/str/int reply
//     never do this; it only matters for cyclic/aliased structures,
//     which a JSON-shaped counts reply cannot contain anyway.
//   - Big integers beyond 64 bits (LONG1/LONG4 opcodes) throw rather
//     than silently truncating.
//   - Protocol-0 textual opcodes (STRING/INT/FLOAT/LONG's text forms)
//     are not supported -- no modern pickler (protocol >= 2) emits them
//     by default, and QMIO's own client pickles at protocol 3.

// Encode the 2-tuple (first, second) -- both plain strings -- as a
// Python pickle, protocol 2. This is QMIO's request shape: (program,
// config_json), both `str`. Protocol 2 is loadable by any Python 3
// (and Python 2) and needs no FRAME/memo machinery for two short
// strings, so the encoder side stays trivial by construction.
std::vector<std::uint8_t> encode_str_tuple(const std::string& first,
                                            const std::string& second);

// Decode a pickle byte stream into a JSON-shaped value: pickle None ->
// null, bool -> bool, int/float -> number, str/bytes -> string (bytes
// decoded as UTF-8; throws if not valid UTF-8, since a counts reply
// should never contain arbitrary binary), list/tuple -> array,
// dict -> object. Throws std::runtime_error on any unsupported or
// unsafe opcode, truncated input, or invalid UTF-8.
nlohmann::json decode(const std::vector<std::uint8_t>& bytes);

} // namespace safe_pickle
} // namespace backends
} // namespace quantum
