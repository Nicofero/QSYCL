#include "safe_pickle.hpp"

#include <cstdio>
#include <cstring>
#include <stdexcept>

namespace quantum {
namespace backends {
namespace safe_pickle {

using json = nlohmann::json;

namespace {

void append_binunicode(std::vector<std::uint8_t>& out, const std::string& s) {
    out.push_back('X'); // BINUNICODE
    std::uint32_t len = static_cast<std::uint32_t>(s.size());
    out.push_back(static_cast<std::uint8_t>(len & 0xFF));
    out.push_back(static_cast<std::uint8_t>((len >> 8) & 0xFF));
    out.push_back(static_cast<std::uint8_t>((len >> 16) & 0xFF));
    out.push_back(static_cast<std::uint8_t>((len >> 24) & 0xFF));
    out.insert(out.end(), s.begin(), s.end());
}

std::string hex_byte(std::uint8_t b) {
    char buf[3];
    std::snprintf(buf, sizeof(buf), "%02x", b);
    return std::string(buf);
}

// Minimal cursor over the reply bytes; every read is bounds-checked so a
// truncated/malformed pickle fails with a clear error instead of
// reading out of bounds.
struct PickleReader {
    const std::uint8_t* data;
    std::size_t size;
    std::size_t pos = 0;

    std::uint8_t read_u8() {
        if (pos >= size) {
            throw std::runtime_error("safe_pickle: truncated input (expected 1 byte)");
        }
        return data[pos++];
    }

    void read_bytes(std::uint8_t* out, std::size_t n) {
        if (pos + n > size) {
            throw std::runtime_error(
                "safe_pickle: truncated input (expected " + std::to_string(n) + " bytes)");
        }
        std::memcpy(out, data + pos, n);
        pos += n;
    }

    std::string read_raw_string(std::size_t n) {
        if (pos + n > size) {
            throw std::runtime_error(
                "safe_pickle: truncated input (expected " + std::to_string(n) +
                "-byte string)");
        }
        std::string s(reinterpret_cast<const char*>(data + pos), n);
        pos += n;
        return s;
    }

    std::uint32_t read_u32_le() {
        std::uint8_t b[4];
        read_bytes(b, 4);
        return static_cast<std::uint32_t>(b[0]) | (static_cast<std::uint32_t>(b[1]) << 8) |
               (static_cast<std::uint32_t>(b[2]) << 16) | (static_cast<std::uint32_t>(b[3]) << 24);
    }

    std::uint64_t read_u64_le() {
        std::uint8_t b[8];
        read_bytes(b, 8);
        std::uint64_t v = 0;
        for (int i = 7; i >= 0; --i) v = (v << 8) | b[i];
        return v;
    }

    std::int32_t read_i32_le() { return static_cast<std::int32_t>(read_u32_le()); }

    std::string read_ascii_line() {
        // PUT/GET (protocol 0/1): ASCII decimal digits terminated by '\n'.
        std::string digits;
        for (;;) {
            char c = static_cast<char>(read_u8());
            if (c == '\n') break;
            digits.push_back(c);
        }
        return digits;
    }
};

} // namespace

std::vector<std::uint8_t> encode_str_tuple(const std::string& first,
                                            const std::string& second) {
    std::vector<std::uint8_t> out;
    out.push_back(0x80); // PROTO
    out.push_back(0x02); // protocol version 2
    append_binunicode(out, first);
    append_binunicode(out, second);
    out.push_back(0x86); // TUPLE2
    out.push_back('.');  // STOP
    return out;
}

json decode(const std::vector<std::uint8_t>& bytes) {
    PickleReader r{bytes.data(), bytes.size()};
    std::vector<json> stack;
    std::vector<std::size_t> marks; // stack indices where MARK was pushed
    std::vector<json> memo;         // PUT/BINPUT/MEMOIZE table (copies -- see header)

    auto pop = [&]() -> json {
        if (stack.empty()) throw std::runtime_error("safe_pickle: stack underflow");
        json v = std::move(stack.back());
        stack.pop_back();
        return v;
    };
    auto pop_mark = [&]() -> std::vector<json> {
        if (marks.empty()) throw std::runtime_error("safe_pickle: MARK stack underflow");
        std::size_t m = marks.back();
        marks.pop_back();
        if (m > stack.size()) throw std::runtime_error("safe_pickle: corrupt mark");
        std::vector<json> items(std::make_move_iterator(stack.begin() + static_cast<long>(m)),
                                 std::make_move_iterator(stack.end()));
        stack.erase(stack.begin() + static_cast<long>(m), stack.end());
        return items;
    };
    auto dict_key = [](const json& k) -> std::string {
        // Pickle dict keys are usually str; anything else (an int key,
        // say) is stringified so it can still be represented as a JSON
        // object key.
        return k.is_string() ? k.get<std::string>() : k.dump();
    };
    auto memo_put = [&](std::size_t index) {
        if (stack.empty()) throw std::runtime_error("safe_pickle: PUT with empty stack");
        if (memo.size() <= index) memo.resize(index + 1);
        memo[index] = stack.back(); // copy -- see header docs
    };
    auto memo_get = [&](std::size_t index) {
        if (index >= memo.size()) {
            throw std::runtime_error(
                "safe_pickle: GET references unknown memo id " + std::to_string(index));
        }
        stack.push_back(memo[index]);
    };

    for (;;) {
        std::uint8_t op = r.read_u8();
        switch (op) {
            case 0x80: // PROTO
                r.read_u8(); // version byte, unchecked -- we support every
                             // opcode any protocol <=4 might emit that we
                             // handle at all, regardless of declared version
                break;
            case 0x95: // FRAME
                r.read_u64_le(); // a buffering hint only, safe to discard
                break;
            case '.': // STOP
                return pop();
            case '(': // MARK
                marks.push_back(stack.size());
                break;
            case '0': // POP
                pop();
                break;
            case '1': // POP_MARK
                pop_mark();
                break;
            case '2': // DUP
                if (stack.empty()) throw std::runtime_error("safe_pickle: DUP with empty stack");
                stack.push_back(stack.back());
                break;
            case 'N': // NONE
                stack.push_back(nullptr);
                break;
            case 0x88: // NEWTRUE
                stack.push_back(true);
                break;
            case 0x89: // NEWFALSE
                stack.push_back(false);
                break;
            case 'J': // BININT (4-byte signed LE)
                stack.push_back(static_cast<std::int64_t>(r.read_i32_le()));
                break;
            case 'K': // BININT1 (1-byte unsigned)
                stack.push_back(static_cast<std::int64_t>(r.read_u8()));
                break;
            case 'M': { // BININT2 (2-byte unsigned LE)
                std::uint8_t b[2];
                r.read_bytes(b, 2);
                stack.push_back(static_cast<std::int64_t>(b[0] | (b[1] << 8)));
                break;
            }
            case 0x8a: { // LONG1
                std::uint8_t n = r.read_u8();
                if (n == 0) {
                    stack.push_back(static_cast<std::int64_t>(0));
                    break;
                }
                if (n > 8) {
                    throw std::runtime_error(
                        "safe_pickle: LONG1 integer wider than 64 bits is not supported");
                }
                std::uint8_t buf[8] = {0};
                r.read_bytes(buf, n);
                std::int64_t v = 0;
                for (int i = n - 1; i >= 0; --i) v = (v << 8) | buf[i];
                if (n < 8 && (buf[n - 1] & 0x80)) {
                    for (int i = n; i < 8; ++i) v |= (static_cast<std::int64_t>(0xFF) << (8 * i));
                }
                stack.push_back(v);
                break;
            }
            case 0x8b: { // LONG4
                std::uint32_t n = r.read_u32_le();
                if (n == 0) {
                    stack.push_back(static_cast<std::int64_t>(0));
                    break;
                }
                if (n > 8) {
                    throw std::runtime_error(
                        "safe_pickle: LONG4 integer wider than 64 bits is not supported");
                }
                std::uint8_t buf[8] = {0};
                r.read_bytes(buf, n);
                std::int64_t v = 0;
                for (int i = static_cast<int>(n) - 1; i >= 0; --i) v = (v << 8) | buf[i];
                if (n < 8 && (buf[n - 1] & 0x80)) {
                    for (std::uint32_t i = n; i < 8; ++i) {
                        v |= (static_cast<std::int64_t>(0xFF) << (8 * i));
                    }
                }
                stack.push_back(v);
                break;
            }
            case 'G': { // BINFLOAT (8-byte BIG-endian IEEE754 double)
                std::uint8_t b[8];
                r.read_bytes(b, 8);
                std::uint64_t bits = 0;
                for (int i = 0; i < 8; ++i) bits = (bits << 8) | b[i];
                double d;
                static_assert(sizeof(double) == 8, "expected IEEE754 double");
                std::memcpy(&d, &bits, 8);
                stack.push_back(d);
                break;
            }
            case 'X': { // BINUNICODE (4-byte LE length + UTF-8 bytes)
                std::uint32_t n = r.read_u32_le();
                stack.push_back(r.read_raw_string(n));
                break;
            }
            case 0x8c: { // SHORT_BINUNICODE (1-byte length + UTF-8 bytes)
                std::uint8_t n = r.read_u8();
                stack.push_back(r.read_raw_string(n));
                break;
            }
            case 0x8d: { // BINUNICODE8 (8-byte LE length + UTF-8 bytes)
                std::uint64_t n = r.read_u64_le();
                stack.push_back(r.read_raw_string(static_cast<std::size_t>(n)));
                break;
            }
            case 'T': { // BINSTRING (4-byte LE length + bytes; protocol <=1 str)
                std::uint32_t n = r.read_u32_le();
                stack.push_back(r.read_raw_string(n));
                break;
            }
            case 'U': { // SHORT_BINSTRING (1-byte length + bytes)
                std::uint8_t n = r.read_u8();
                stack.push_back(r.read_raw_string(n));
                break;
            }
            case 'B': { // BINBYTES (4-byte LE length + raw bytes)
                std::uint32_t n = r.read_u32_le();
                stack.push_back(r.read_raw_string(n)); // best-effort as text
                break;
            }
            case 'C': { // SHORT_BINBYTES (1-byte length + raw bytes)
                std::uint8_t n = r.read_u8();
                stack.push_back(r.read_raw_string(n));
                break;
            }
            case 0x8e: { // BINBYTES8 (8-byte LE length + raw bytes)
                std::uint64_t n = r.read_u64_le();
                stack.push_back(r.read_raw_string(static_cast<std::size_t>(n)));
                break;
            }
            case '}': // EMPTY_DICT
                stack.push_back(json::object());
                break;
            case ']': // EMPTY_LIST
                stack.push_back(json::array());
                break;
            case ')': // EMPTY_TUPLE
                stack.push_back(json::array());
                break;
            case 0x85: { // TUPLE1
                json a = pop();
                stack.push_back(json::array({std::move(a)}));
                break;
            }
            case 0x86: { // TUPLE2
                json b = pop();
                json a = pop();
                stack.push_back(json::array({std::move(a), std::move(b)}));
                break;
            }
            case 0x87: { // TUPLE3
                json c = pop();
                json b = pop();
                json a = pop();
                stack.push_back(json::array({std::move(a), std::move(b), std::move(c)}));
                break;
            }
            case 't': { // TUPLE (mark-delimited)
                auto items = pop_mark();
                stack.push_back(json(items));
                break;
            }
            case 's': { // SETITEM
                json value = pop();
                json key = pop();
                if (stack.empty() || !stack.back().is_object()) {
                    throw std::runtime_error("safe_pickle: SETITEM without a dict on the stack");
                }
                stack.back()[dict_key(key)] = std::move(value);
                break;
            }
            case 'u': { // SETITEMS
                auto items = pop_mark();
                if (stack.empty() || !stack.back().is_object()) {
                    throw std::runtime_error("safe_pickle: SETITEMS without a dict on the stack");
                }
                if (items.size() % 2 != 0) {
                    throw std::runtime_error("safe_pickle: SETITEMS with an odd number of items");
                }
                for (std::size_t i = 0; i < items.size(); i += 2) {
                    stack.back()[dict_key(items[i])] = items[i + 1];
                }
                break;
            }
            case 'a': { // APPEND
                json value = pop();
                if (stack.empty() || !stack.back().is_array()) {
                    throw std::runtime_error("safe_pickle: APPEND without a list on the stack");
                }
                stack.back().push_back(std::move(value));
                break;
            }
            case 'e': { // APPENDS
                auto items = pop_mark();
                if (stack.empty() || !stack.back().is_array()) {
                    throw std::runtime_error("safe_pickle: APPENDS without a list on the stack");
                }
                for (auto& v : items) stack.back().push_back(std::move(v));
                break;
            }
            case 'p': // PUT (protocol 0/1)
                memo_put(static_cast<std::size_t>(std::stoull(r.read_ascii_line())));
                break;
            case 'q': // BINPUT (1-byte index)
                memo_put(r.read_u8());
                break;
            case 'r': // LONG_BINPUT (4-byte LE index)
                memo_put(r.read_u32_le());
                break;
            case 0x94: // MEMOIZE (implicit next index)
                memo_put(memo.size());
                break;
            case 'g': // GET (protocol 0/1)
                memo_get(static_cast<std::size_t>(std::stoull(r.read_ascii_line())));
                break;
            case 'h': // BINGET (1-byte index)
                memo_get(r.read_u8());
                break;
            case 'j': // LONG_BINGET (4-byte LE index)
                memo_get(r.read_u32_le());
                break;
            // Explicitly refused: object construction / code execution.
            // See the safety note in safe_pickle.hpp -- do not add these.
            case 'R':   // REDUCE
            case 'c':   // GLOBAL
            case 0x93:  // STACK_GLOBAL
            case 'b':   // BUILD
            case 'i':   // INST
            case 'o':   // OBJ
            case 0x81:  // NEWOBJ
            case 0x92:  // NEWOBJ_EX
            case 0x82:  // EXT1
            case 0x83:  // EXT2
            case 0x84:  // EXT4
            case 'P':   // PERSID
            case 'Q':   // BINPERSID
                throw std::runtime_error(
                    "safe_pickle: refusing unsafe opcode 0x" + hex_byte(op) +
                    " (object construction / code execution opcodes are never interpreted)");
            default:
                throw std::runtime_error("safe_pickle: unsupported opcode 0x" + hex_byte(op));
        }
    }
}

} // namespace safe_pickle
} // namespace backends
} // namespace quantum
