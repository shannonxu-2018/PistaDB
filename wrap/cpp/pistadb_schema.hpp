/**
 *    ___ _    _        ___  ___
 *   | _ (_)__| |_ __ _|   \| _ )
 *   |  _/ (_-<  _/ _` | |) | _ \
 *   |_| |_/__/\__\__,_|___/|___/
 *
 * pistadb_schema.hpp — Milvus-style schema layer for the C++ wrapper.
 *
 * Header-only, depends only on the C++17 standard library (no nlohmann/json,
 * RapidJSON, etc.).  A minimal JSON writer/parser is bundled below and only
 * supports the subset of JSON we actually emit for the sidecar.
 *
 * Quick start:
 *   #include "pistadb.hpp"
 *   #include "pistadb_schema.hpp"
 *
 *   using namespace pistadb;
 *   std::vector<FieldSchema> fields = {
 *       {"lc_id",     DataType::Int64,        true, true},
 *       {"lc_section",DataType::VarChar,      false,false, 100},
 *       {"lc_vector", DataType::FloatVector,  false,false, std::nullopt, 1536},
 *   };
 *   auto coll = create_collection(
 *       "common_text", fields, "Common text search",
 *       { Metric::Cosine, IndexType::HNSW, std::nullopt, std::string("./db") });
 *
 *   coll.insert({{
 *       {"lc_section", Value::str("common")},
 *       {"lc_vector",  Value::floats(vec_1536)},
 *   }});
 *   auto hits = coll.search(query, 10);
 *   coll.flush();
 */
#pragma once

#include "pistadb.hpp"

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <fstream>
#include <map>
#include <memory>
#include <optional>
#include <sstream>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <variant>
#include <vector>

namespace pistadb {

// ── DataType ──────────────────────────────────────────────────────────────────

enum class DataType : int {
    Bool        = 1,
    Int8        = 2,
    Int16       = 3,
    Int32       = 4,
    Int64       = 5,
    Float       = 10,
    Double      = 11,
    VarChar     = 21,
    Json        = 23,
    FloatVector = 101,
};

inline const char* to_wire(DataType d) {
    switch (d) {
        case DataType::Bool:        return "BOOL";
        case DataType::Int8:        return "INT8";
        case DataType::Int16:       return "INT16";
        case DataType::Int32:       return "INT32";
        case DataType::Int64:       return "INT64";
        case DataType::Float:       return "FLOAT";
        case DataType::Double:      return "DOUBLE";
        case DataType::VarChar:     return "VARCHAR";
        case DataType::Json:        return "JSON";
        case DataType::FloatVector: return "FLOAT_VECTOR";
    }
    return "?";
}

inline DataType from_wire(const std::string& s) {
    if (s == "BOOL")         return DataType::Bool;
    if (s == "INT8")         return DataType::Int8;
    if (s == "INT16")        return DataType::Int16;
    if (s == "INT32")        return DataType::Int32;
    if (s == "INT64")        return DataType::Int64;
    if (s == "FLOAT")        return DataType::Float;
    if (s == "DOUBLE")       return DataType::Double;
    if (s == "VARCHAR")      return DataType::VarChar;
    if (s == "JSON")         return DataType::Json;
    if (s == "FLOAT_VECTOR") return DataType::FloatVector;
    throw Exception("unknown DataType: " + s);
}

inline bool is_int(DataType d) {
    return d == DataType::Int8 || d == DataType::Int16 ||
           d == DataType::Int32 || d == DataType::Int64;
}

inline bool is_float(DataType d) {
    return d == DataType::Float || d == DataType::Double;
}

// ── Value (variant for scalar / vector cells) ─────────────────────────────────

class Value {
public:
    enum class Kind { Null, Bool, Int, Double, String, Floats };

    Value() : kind_(Kind::Null) {}

    static Value null()                      { return Value(); }
    static Value boolean(bool b)             { Value v; v.kind_=Kind::Bool;   v.b_=b;             return v; }
    static Value integer(int64_t i)          { Value v; v.kind_=Kind::Int;    v.i_=i;             return v; }
    static Value real(double d)              { Value v; v.kind_=Kind::Double; v.d_=d;             return v; }
    static Value str(std::string s)          { Value v; v.kind_=Kind::String; v.s_=std::move(s);  return v; }
    static Value floats(std::vector<float> f){ Value v; v.kind_=Kind::Floats; v.f_=std::move(f);  return v; }

    Kind kind() const noexcept { return kind_; }

    bool                       is_null()   const { return kind_ == Kind::Null;   }
    bool                       as_bool()   const { check(Kind::Bool);   return b_;}
    int64_t                    as_int()    const { check(Kind::Int);    return i_;}
    double                     as_double() const { check(Kind::Double); return d_;}
    const std::string&         as_string() const { check(Kind::String); return s_;}
    const std::vector<float>&  as_floats() const { check(Kind::Floats); return f_;}

private:
    void check(Kind expected) const {
        if (kind_ != expected) throw Exception("Value: type mismatch");
    }

    Kind                kind_;
    bool                b_ = false;
    int64_t             i_ = 0;
    double              d_ = 0.0;
    std::string         s_;
    std::vector<float>  f_;
};

// ── FieldSchema ───────────────────────────────────────────────────────────────

struct FieldSchema {
    std::string         name;
    DataType            dtype       = DataType::Int64;
    bool                is_primary  = false;
    bool                auto_id     = false;
    std::optional<int>  max_length; // VARCHAR only
    std::optional<int>  dim;        // FLOAT_VECTOR only
    std::string         description;

    void validate() const {
        if (name.empty())
            throw Exception("FieldSchema: name must not be empty");
        if (dtype == DataType::FloatVector && (!dim || *dim <= 0))
            throw Exception("FLOAT_VECTOR field '" + name + "' requires positive dim");
        if (dtype == DataType::VarChar && max_length && *max_length <= 0)
            throw Exception("VARCHAR field '" + name + "': max_length must be positive");
        if (is_primary && dtype != DataType::Int64)
            throw Exception("primary key '" + name + "' must be INT64");
        if (auto_id && !is_primary)
            throw Exception("auto_id only valid on primary field (got '" + name + "')");
    }
};

// ── CollectionSchema ──────────────────────────────────────────────────────────

class CollectionSchema {
public:
    std::vector<FieldSchema> fields;
    std::string              description;

    CollectionSchema(std::vector<FieldSchema> fs, std::string desc = "")
        : fields(std::move(fs)), description(std::move(desc))
    {
        validate();
        for (size_t i = 0; i < fields.size(); ++i) {
            if (fields[i].is_primary)              primary_idx_ = i;
            if (fields[i].dtype == DataType::FloatVector) vector_idx_ = i;
        }
    }

    const FieldSchema& primary_field() const { return fields[primary_idx_]; }
    const FieldSchema& vector_field()  const { return fields[vector_idx_];  }

    std::vector<const FieldSchema*> scalar_fields() const {
        std::vector<const FieldSchema*> out;
        for (const auto& f : fields)
            if (!f.is_primary && f.dtype != DataType::FloatVector)
                out.push_back(&f);
        return out;
    }

    const FieldSchema* find(const std::string& name) const {
        for (const auto& f : fields) if (f.name == name) return &f;
        return nullptr;
    }

private:
    void validate() {
        std::unordered_set<std::string> seen;
        int n_primary = 0, n_vector = 0;
        for (auto& f : fields) {
            f.validate();
            if (!seen.insert(f.name).second)
                throw Exception("duplicate field name '" + f.name + "'");
            if (f.is_primary) ++n_primary;
            if (f.dtype == DataType::FloatVector) ++n_vector;
        }
        if (n_primary != 1)
            throw Exception("schema must have exactly one primary key (found " +
                            std::to_string(n_primary) + ")");
        if (n_vector != 1)
            throw Exception("schema must have exactly one FLOAT_VECTOR field (found " +
                            std::to_string(n_vector) + ")");
    }

    size_t primary_idx_ = 0;
    size_t vector_idx_  = 0;
};

// ── Hit ───────────────────────────────────────────────────────────────────────

struct Hit {
    uint64_t                       id;
    float                          distance;
    std::map<std::string, Value>   fields;

    const Value* get(const std::string& name) const {
        auto it = fields.find(name);
        return it == fields.end() ? nullptr : &it->second;
    }
};

// ── Mini-JSON writer / parser ────────────────────────────────────────────────
namespace detail {

// ── Writer ────────────────────────────────────────────────────────────────────
class JsonWriter {
public:
    explicit JsonWriter(std::string& out) : out_(out) {}

    void value_null()              { out_ += "null"; }
    void value_bool(bool b)        { out_ += b ? "true" : "false"; }
    void value_int(int64_t i)      { out_ += std::to_string(i); }
    void value_double(double d) {
        std::ostringstream os;
        os.precision(17);
        os << d;
        out_ += os.str();
    }
    void value_string(const std::string& s) { write_string(s); }

    void key(const std::string& k)  { write_string(k); out_ += ':'; }

    void object_begin() { out_ += '{'; first_.push_back(true); }
    void object_end()   { out_ += '}'; first_.pop_back();        }

    void array_begin()  { out_ += '['; first_.push_back(true); }
    void array_end()    { out_ += ']'; first_.pop_back();        }

    void comma_if_needed() {
        if (first_.empty()) return;
        if (first_.back()) first_.back() = false;
        else               out_ += ',';
    }

    void write_value(const Value& v) {
        comma_if_needed();
        switch (v.kind()) {
            case Value::Kind::Null:   value_null(); break;
            case Value::Kind::Bool:   value_bool(v.as_bool()); break;
            case Value::Kind::Int:    value_int(v.as_int()); break;
            case Value::Kind::Double: value_double(v.as_double()); break;
            case Value::Kind::String: value_string(v.as_string()); break;
            case Value::Kind::Floats: {
                array_begin();
                for (float f : v.as_floats()) {
                    comma_if_needed();
                    value_double(static_cast<double>(f));
                }
                array_end();
                break;
            }
        }
    }

private:
    void write_string(const std::string& s) {
        out_ += '"';
        for (char c : s) {
            switch (c) {
                case '"':  out_ += "\\\""; break;
                case '\\': out_ += "\\\\"; break;
                case '\n': out_ += "\\n";  break;
                case '\r': out_ += "\\r";  break;
                case '\t': out_ += "\\t";  break;
                case '\b': out_ += "\\b";  break;
                case '\f': out_ += "\\f";  break;
                default:
                    if (static_cast<unsigned char>(c) < 0x20) {
                        char buf[8];
                        std::snprintf(buf, sizeof buf, "\\u%04x", c);
                        out_ += buf;
                    } else {
                        out_ += c;
                    }
            }
        }
        out_ += '"';
    }

    std::string&        out_;
    std::vector<bool>   first_;
};

// ── Parser ─────────────────────────────────────────────────────────────────────
// Returns a small JSON value tree.

struct JsonValue {
    enum class Tag { Null, Bool, Int, Double, String, Array, Object };
    Tag tag = Tag::Null;
    bool   b = false;
    int64_t i = 0;
    double d = 0.0;
    std::string s;
    std::vector<JsonValue> arr;
    std::map<std::string, JsonValue> obj;
};

class JsonParser {
public:
    explicit JsonParser(const std::string& src) : src_(src), pos_(0) {}

    JsonValue parse() {
        skip_ws();
        auto v = parse_value();
        skip_ws();
        if (pos_ != src_.size()) throw Exception("JSON: trailing data");
        return v;
    }

private:
    JsonValue parse_value() {
        skip_ws();
        if (pos_ >= src_.size()) throw Exception("JSON: unexpected EOF");
        char c = src_[pos_];
        if (c == '{') return parse_object();
        if (c == '[') return parse_array();
        if (c == '"') return parse_string_value();
        if (c == 't' || c == 'f') return parse_bool();
        if (c == 'n') return parse_null();
        if (c == '-' || (c >= '0' && c <= '9')) return parse_number();
        throw Exception("JSON: unexpected char '" + std::string(1, c) + "'");
    }

    JsonValue parse_object() {
        JsonValue v; v.tag = JsonValue::Tag::Object;
        ++pos_;                       // '{'
        skip_ws();
        if (peek() == '}') { ++pos_; return v; }
        while (true) {
            skip_ws();
            std::string k = parse_string_raw();
            skip_ws();
            expect(':');
            auto child = parse_value();
            v.obj.emplace(std::move(k), std::move(child));
            skip_ws();
            char c = peek();
            if (c == ',') { ++pos_; continue; }
            if (c == '}') { ++pos_; return v; }
            throw Exception("JSON: expected ',' or '}'");
        }
    }

    JsonValue parse_array() {
        JsonValue v; v.tag = JsonValue::Tag::Array;
        ++pos_;                       // '['
        skip_ws();
        if (peek() == ']') { ++pos_; return v; }
        while (true) {
            v.arr.push_back(parse_value());
            skip_ws();
            char c = peek();
            if (c == ',') { ++pos_; continue; }
            if (c == ']') { ++pos_; return v; }
            throw Exception("JSON: expected ',' or ']'");
        }
    }

    JsonValue parse_string_value() {
        JsonValue v; v.tag = JsonValue::Tag::String;
        v.s = parse_string_raw();
        return v;
    }

    JsonValue parse_bool() {
        if (src_.compare(pos_, 4, "true") == 0)  { pos_ += 4; JsonValue v; v.tag = JsonValue::Tag::Bool; v.b = true;  return v; }
        if (src_.compare(pos_, 5, "false") == 0) { pos_ += 5; JsonValue v; v.tag = JsonValue::Tag::Bool; v.b = false; return v; }
        throw Exception("JSON: bad bool");
    }

    JsonValue parse_null() {
        if (src_.compare(pos_, 4, "null") == 0) { pos_ += 4; return JsonValue{}; }
        throw Exception("JSON: bad null");
    }

    JsonValue parse_number() {
        size_t start = pos_;
        if (src_[pos_] == '-') ++pos_;
        bool is_double = false;
        while (pos_ < src_.size()) {
            char c = src_[pos_];
            if (c == '.' || c == 'e' || c == 'E') is_double = true;
            if ((c >= '0' && c <= '9') || c == '.' || c == 'e' || c == 'E' || c == '+' || c == '-') ++pos_;
            else break;
        }
        std::string num = src_.substr(start, pos_ - start);
        JsonValue v;
        if (is_double) {
            v.tag = JsonValue::Tag::Double;
            v.d   = std::stod(num);
        } else {
            v.tag = JsonValue::Tag::Int;
            v.i   = std::stoll(num);
        }
        return v;
    }

    std::string parse_string_raw() {
        if (peek() != '"') throw Exception("JSON: expected '\"'");
        ++pos_;
        std::string out;
        while (pos_ < src_.size()) {
            char c = src_[pos_++];
            if (c == '"') return out;
            if (c == '\\') {
                if (pos_ >= src_.size()) throw Exception("JSON: bad escape");
                char esc = src_[pos_++];
                switch (esc) {
                    case '"':  out += '"';  break;
                    case '\\': out += '\\'; break;
                    case '/':  out += '/';  break;
                    case 'n':  out += '\n'; break;
                    case 'r':  out += '\r'; break;
                    case 't':  out += '\t'; break;
                    case 'b':  out += '\b'; break;
                    case 'f':  out += '\f'; break;
                    case 'u': {
                        if (pos_ + 4 > src_.size()) throw Exception("JSON: bad \\u");
                        unsigned code = std::stoul(src_.substr(pos_, 4), nullptr, 16);
                        pos_ += 4;
                        // emit UTF-8 (BMP only — surrogate pairs unsupported in sidecar)
                        if (code < 0x80) {
                            out += static_cast<char>(code);
                        } else if (code < 0x800) {
                            out += static_cast<char>(0xC0 | (code >> 6));
                            out += static_cast<char>(0x80 | (code & 0x3F));
                        } else {
                            out += static_cast<char>(0xE0 | (code >> 12));
                            out += static_cast<char>(0x80 | ((code >> 6) & 0x3F));
                            out += static_cast<char>(0x80 | (code & 0x3F));
                        }
                        break;
                    }
                    default: throw Exception("JSON: unknown escape");
                }
            } else {
                out += c;
            }
        }
        throw Exception("JSON: unterminated string");
    }

    void skip_ws() {
        while (pos_ < src_.size()) {
            char c = src_[pos_];
            if (c == ' ' || c == '\t' || c == '\n' || c == '\r') ++pos_;
            else break;
        }
    }

    char peek() const {
        if (pos_ >= src_.size()) throw Exception("JSON: unexpected EOF");
        return src_[pos_];
    }

    void expect(char c) {
        if (peek() != c) throw Exception(std::string("JSON: expected '") + c + "'");
        ++pos_;
    }

    const std::string& src_;
    size_t pos_;
};

inline Value json_to_value(const JsonValue& jv) {
    using Tag = JsonValue::Tag;
    switch (jv.tag) {
        case Tag::Null:   return Value::null();
        case Tag::Bool:   return Value::boolean(jv.b);
        case Tag::Int:    return Value::integer(jv.i);
        case Tag::Double: return Value::real(jv.d);
        case Tag::String: return Value::str(jv.s);
        case Tag::Array: {
            std::vector<float> floats;
            floats.reserve(jv.arr.size());
            for (const auto& e : jv.arr) {
                if      (e.tag == Tag::Double) floats.push_back(static_cast<float>(e.d));
                else if (e.tag == Tag::Int)    floats.push_back(static_cast<float>(e.i));
                else                            return Value::null(); // mixed/non-numeric → unsupported in cell
            }
            return Value::floats(std::move(floats));
        }
        case Tag::Object:
            // Nested objects are not supported in the schema layer; ignore.
            return Value::null();
    }
    return Value::null();
}

} // namespace detail

// ── CollectionOptions ─────────────────────────────────────────────────────────

struct CollectionOptions {
    Metric                       metric     = Metric::L2;
    IndexType                    index_type = IndexType::HNSW;
    std::optional<Params>        params;
    std::optional<std::string>   base_dir;   // ignored if `path` is set
    std::optional<std::string>   path;       // explicit .pst path
    bool                         overwrite  = false;
};

namespace detail {

inline std::string resolve_path(const std::string& name,
                                const CollectionOptions& opt)
{
    if (opt.path) return *opt.path;
    if (opt.base_dir) {
        const auto& d = *opt.base_dir;
        if (d.empty()) return name + ".pst";
        char sep = (d.find('\\') != std::string::npos && d.find('/') == std::string::npos) ? '\\' : '/';
        if (d.back() == '/' || d.back() == '\\') return d + name + ".pst";
        return d + sep + name + ".pst";
    }
    return name + ".pst";
}

inline const char* metric_name(Metric m) {
    switch (m) {
        case Metric::L2:      return "L2";
        case Metric::Cosine:  return "Cosine";
        case Metric::IP:      return "IP";
        case Metric::L1:      return "L1";
        case Metric::Hamming: return "Hamming";
    }
    return "L2";
}

inline Metric metric_from_name(const std::string& s) {
    if (s == "L2")           return Metric::L2;
    if (s == "Cosine")       return Metric::Cosine;
    if (s == "IP" || s == "InnerProduct") return Metric::IP;
    if (s == "L1")           return Metric::L1;
    if (s == "Hamming")      return Metric::Hamming;
    throw Exception("unknown metric: " + s);
}

inline const char* index_name(IndexType i) {
    switch (i) {
        case IndexType::Linear:  return "Linear";
        case IndexType::HNSW:    return "HNSW";
        case IndexType::IVF:     return "IVF";
        case IndexType::IVF_PQ:  return "IVF_PQ";
        case IndexType::DiskANN: return "DiskANN";
        case IndexType::LSH:     return "LSH";
        case IndexType::ScaNN:   return "ScaNN";
    }
    return "HNSW";
}

inline IndexType index_from_name(const std::string& s) {
    if (s == "Linear")  return IndexType::Linear;
    if (s == "HNSW")    return IndexType::HNSW;
    if (s == "IVF")     return IndexType::IVF;
    if (s == "IVF_PQ")  return IndexType::IVF_PQ;
    if (s == "DiskANN") return IndexType::DiskANN;
    if (s == "LSH")     return IndexType::LSH;
    if (s == "ScaNN")   return IndexType::ScaNN;
    throw Exception("unknown index type: " + s);
}

inline bool file_exists(const std::string& p) {
    std::ifstream f(p);
    return f.good();
}

inline std::string read_text_file(const std::string& p) {
    std::ifstream f(p, std::ios::binary);
    if (!f) throw Exception("cannot read " + p);
    std::ostringstream os; os << f.rdbuf();
    return os.str();
}

inline void write_text_file_atomic(const std::string& p, const std::string& s) {
    const std::string tmp = p + ".tmp";
    {
        std::ofstream f(tmp, std::ios::binary | std::ios::trunc);
        if (!f) throw Exception("cannot write " + tmp);
        f.write(s.data(), static_cast<std::streamsize>(s.size()));
        if (!f) throw Exception("write failed: " + tmp);
    }
    std::remove(p.c_str());
    if (std::rename(tmp.c_str(), p.c_str()) != 0)
        throw Exception("rename failed: " + tmp + " -> " + p);
}

inline Value coerce_scalar(const Value& v, const FieldSchema& f) {
    if (v.is_null()) return Value::null();
    switch (f.dtype) {
        case DataType::Bool:
            if (v.kind() == Value::Kind::Bool) return v;
            throw Exception("field " + f.name + ": expected bool");
        case DataType::VarChar: {
            if (v.kind() != Value::Kind::String)
                throw Exception("field " + f.name + ": expected string");
            const auto& s = v.as_string();
            if (f.max_length && static_cast<int>(s.size()) > *f.max_length)
                throw Exception("field " + f.name + ": exceeds max_length");
            return v;
        }
        case DataType::Json:
            // Accept any value as JSON; serialise to its native kind.
            return v;
        default:
            if (is_int(f.dtype)) {
                if (v.kind() == Value::Kind::Int)    return v;
                if (v.kind() == Value::Kind::Double) return Value::integer(static_cast<int64_t>(v.as_double()));
                throw Exception("field " + f.name + ": expected integer");
            }
            if (is_float(f.dtype)) {
                if (v.kind() == Value::Kind::Double) return v;
                if (v.kind() == Value::Kind::Int)    return Value::real(static_cast<double>(v.as_int()));
                throw Exception("field " + f.name + ": expected number");
            }
            throw Exception("field " + f.name + ": unsupported dtype");
    }
}

} // namespace detail

// ── Collection ────────────────────────────────────────────────────────────────

class Collection {
public:
    using Row = std::map<std::string, Value>;

    Collection(Collection&&) noexcept = default;
    Collection& operator=(Collection&&) noexcept = default;
    Collection(const Collection&)            = delete;
    Collection& operator=(const Collection&) = delete;

    const std::string&       name()        const { return name_; }
    const std::string&       path()        const { return path_; }
    const CollectionSchema&  schema()      const { return schema_; }
    Database&                database()         { return *db_;  }
    int                      num_entities() const { return db_->count(); }

    /// Persist both the .pst file and the JSON sidecar.
    void flush() {
        db_->save();
        save_sidecar();
    }

    /// Alias for flush().
    void save() { flush(); }

    /// Insert one or more rows.  Returns the assigned primary ids.
    std::vector<uint64_t> insert(const std::vector<Row>& rows) {
        const auto& pk      = schema_.primary_field();
        const auto& vec     = schema_.vector_field();
        const auto  scalars = schema_.scalar_fields();

        std::unordered_set<std::string> known;
        for (const auto& f : schema_.fields) known.insert(f.name);

        std::vector<uint64_t> ids;
        ids.reserve(rows.size());
        for (const auto& row : rows) {
            for (const auto& kv : row)
                if (!known.count(kv.first))
                    throw Exception("unknown field '" + kv.first + "'");

            uint64_t id;
            auto pk_it = row.find(pk.name);
            if (pk.auto_id) {
                if (pk_it != row.end() && !pk_it->second.is_null())
                    throw Exception("auto_id enabled on '" + pk.name + "' — do not supply it");
                id = next_id_++;
            } else {
                if (pk_it == row.end() || pk_it->second.is_null())
                    throw Exception("missing primary key '" + pk.name + "'");
                if (pk_it->second.kind() != Value::Kind::Int)
                    throw Exception("primary key must be integer");
                int64_t signed_id = pk_it->second.as_int();
                if (signed_id <= 0) throw Exception("primary key must be > 0");
                id = static_cast<uint64_t>(signed_id);
                if (rows_.count(id))
                    throw Exception("duplicate primary id=" + std::to_string(id));
                if (id >= next_id_) next_id_ = id + 1;
            }

            auto vec_it = row.find(vec.name);
            if (vec_it == row.end() || vec_it->second.kind() != Value::Kind::Floats)
                throw Exception("missing or invalid vector field '" + vec.name + "'");
            const auto& v = vec_it->second.as_floats();
            if (static_cast<int>(v.size()) != *vec.dim)
                throw Exception("vector length mismatch for '" + vec.name + "'");

            Row scalar_vals;
            for (auto* f : scalars) {
                auto it = row.find(f->name);
                scalar_vals[f->name] = (it == row.end() || it->second.is_null())
                    ? Value::null()
                    : detail::coerce_scalar(it->second, *f);
            }

            db_->insert(id, v.data());
            rows_[id] = std::move(scalar_vals);
            ids.push_back(id);
        }
        return ids;
    }

    /// Delete by primary id.  Missing ids are skipped silently; returns the
    /// number of rows actually removed.
    int remove(const std::vector<uint64_t>& ids) {
        int removed = 0;
        for (auto id : ids) {
            try {
                db_->remove(id);
                rows_.erase(id);
                ++removed;
            } catch (const Exception&) { /* ignore */ }
        }
        return removed;
    }

    /// Get the full row (all fields, including the vector) for the given id.
    Row get(uint64_t id) const {
        auto it = rows_.find(id);
        if (it == rows_.end()) throw Exception("id=" + std::to_string(id) + " not found");
        auto entry = db_->get(id);
        Row out = it->second;
        out[schema_.primary_field().name] = Value::integer(static_cast<int64_t>(id));
        out[schema_.vector_field().name]  = Value::floats(std::move(entry.vector));
        return out;
    }

    /// k-NN search.  Pass an empty `output_fields` to project all scalar fields.
    std::vector<Hit> search(const std::vector<float>& query,
                            int k,
                            const std::vector<std::string>& output_fields = {}) const
    {
        const auto& vec = schema_.vector_field();
        if (static_cast<int>(query.size()) != *vec.dim)
            throw Exception("query size mismatch");

        std::vector<std::string> want;
        if (output_fields.empty()) {
            for (auto* f : schema_.scalar_fields()) want.push_back(f->name);
        } else {
            const auto pk_name = schema_.primary_field().name;
            for (const auto& n : output_fields) {
                if (n != pk_name && n != vec.name && !schema_.find(n))
                    throw Exception("unknown output field '" + n + "'");
                want.push_back(n);
            }
        }

        auto raw = db_->search(query, k);
        const std::string& pk_name  = schema_.primary_field().name;
        const std::string& vec_name = vec.name;

        std::vector<Hit> hits;
        hits.reserve(raw.size());
        for (const auto& r : raw) {
            Hit h{ r.id, r.distance, {} };
            auto rit = rows_.find(r.id);
            for (const auto& n : want) {
                if (n == pk_name) {
                    h.fields[n] = Value::integer(static_cast<int64_t>(r.id));
                } else if (n == vec_name) {
                    auto entry = db_->get(r.id);
                    h.fields[n] = Value::floats(std::move(entry.vector));
                } else if (rit != rows_.end()) {
                    auto fit = rit->second.find(n);
                    h.fields[n] = (fit != rit->second.end()) ? fit->second : Value::null();
                } else {
                    h.fields[n] = Value::null();
                }
            }
            hits.push_back(std::move(h));
        }
        return hits;
    }

    // ── Internal: factories use these ─────────────────────────────────────
    static Collection make_(std::string name,
                            std::string path,
                            CollectionSchema schema,
                            std::unique_ptr<Database> db,
                            Metric metric,
                            IndexType idx,
                            std::map<uint64_t, Row> rows,
                            uint64_t next_id)
    {
        Collection c;
        c.name_    = std::move(name);
        c.path_    = std::move(path);
        c.meta_    = c.path_ + ".meta.json";
        c.schema_  = std::move(schema);
        c.db_      = std::move(db);
        c.metric_  = metric;
        c.index_   = idx;
        c.rows_    = std::move(rows);
        c.next_id_ = next_id;
        return c;
    }

    void save_sidecar() {
        std::string out;
        detail::JsonWriter w(out);
        w.object_begin();
        w.comma_if_needed(); w.key("version");     w.value_int(1);
        w.comma_if_needed(); w.key("name");        w.value_string(name_);
        w.comma_if_needed(); w.key("description"); w.value_string(schema_.description);
        w.comma_if_needed(); w.key("metric");      w.value_string(detail::metric_name(metric_));
        w.comma_if_needed(); w.key("index");       w.value_string(detail::index_name(index_));
        w.comma_if_needed(); w.key("next_id");     w.value_int(static_cast<int64_t>(next_id_));

        w.comma_if_needed(); w.key("fields");
        w.array_begin();
        for (const auto& f : schema_.fields) {
            w.comma_if_needed();
            w.object_begin();
            w.comma_if_needed(); w.key("name");        w.value_string(f.name);
            w.comma_if_needed(); w.key("dtype");       w.value_string(to_wire(f.dtype));
            w.comma_if_needed(); w.key("is_primary");  w.value_bool(f.is_primary);
            w.comma_if_needed(); w.key("auto_id");     w.value_bool(f.auto_id);
            if (f.max_length) { w.comma_if_needed(); w.key("max_length"); w.value_int(*f.max_length); }
            if (f.dim)        { w.comma_if_needed(); w.key("dim");        w.value_int(*f.dim); }
            if (!f.description.empty()) {
                w.comma_if_needed(); w.key("description"); w.value_string(f.description);
            }
            w.object_end();
        }
        w.array_end();

        w.comma_if_needed(); w.key("rows");
        w.object_begin();
        for (const auto& kv : rows_) {
            w.comma_if_needed();
            w.key(std::to_string(kv.first));
            w.object_begin();
            for (const auto& col : kv.second) {
                w.comma_if_needed();
                w.key(col.first);
                w.write_value(col.second);
            }
            w.object_end();
        }
        w.object_end();

        w.object_end();
        detail::write_text_file_atomic(meta_, out);
    }

private:
    Collection() = default;

    std::string                     name_;
    std::string                     path_;
    std::string                     meta_;
    CollectionSchema                schema_{ {{"_pk_placeholder", DataType::Int64, true},
                                             {"_v_placeholder", DataType::FloatVector, false, false, std::nullopt, 1}} };
    std::unique_ptr<Database>       db_;
    Metric                          metric_ = Metric::L2;
    IndexType                       index_  = IndexType::HNSW;
    std::map<uint64_t, Row>         rows_;
    uint64_t                        next_id_ = 1;
};

// ── Factories ────────────────────────────────────────────────────────────────

inline Collection create_collection(
    const std::string& name,
    std::vector<FieldSchema> fields,
    const std::string& description = "",
    const CollectionOptions& opt = {})
{
    CollectionSchema schema(std::move(fields), description);
    auto path = detail::resolve_path(name, opt);
    auto meta = path + ".meta.json";

    if (opt.overwrite) {
        std::remove(path.c_str());
        std::remove(meta.c_str());
    } else {
        if (detail::file_exists(path)) throw Exception(path + " already exists");
        if (detail::file_exists(meta)) throw Exception(meta + " already exists");
    }

    auto db = std::make_unique<Database>(
        path,
        *schema.vector_field().dim,
        opt.metric,
        opt.index_type,
        opt.params ? &(*opt.params) : nullptr);

    auto coll = Collection::make_(
        name, path, std::move(schema), std::move(db),
        opt.metric, opt.index_type, {}, 1);
    coll.save_sidecar();
    return coll;
}

inline Collection load_collection(
    const std::string& name,
    const CollectionOptions& opt = {})
{
    auto path = detail::resolve_path(name, opt);
    auto meta = path + ".meta.json";
    if (!detail::file_exists(meta))
        throw Exception("sidecar not found: " + meta);

    auto raw = detail::read_text_file(meta);
    auto jv  = detail::JsonParser(raw).parse();
    if (jv.tag != detail::JsonValue::Tag::Object)
        throw Exception("sidecar: expected object");

    auto& obj = jv.obj;

    auto get_str = [&](const char* k) {
        auto it = obj.find(k);
        if (it == obj.end() || it->second.tag != detail::JsonValue::Tag::String)
            throw Exception(std::string("sidecar: missing string field '") + k + "'");
        return it->second.s;
    };
    auto opt_str = [&](const char* k) -> std::string {
        auto it = obj.find(k);
        if (it == obj.end() || it->second.tag != detail::JsonValue::Tag::String) return {};
        return it->second.s;
    };

    // Fields
    std::vector<FieldSchema> fields;
    {
        auto fit = obj.find("fields");
        if (fit == obj.end() || fit->second.tag != detail::JsonValue::Tag::Array)
            throw Exception("sidecar: missing 'fields' array");
        for (const auto& je : fit->second.arr) {
            if (je.tag != detail::JsonValue::Tag::Object)
                throw Exception("sidecar: field entry not an object");
            FieldSchema f;
            for (const auto& kv : je.obj) {
                if      (kv.first == "name")        f.name        = kv.second.s;
                else if (kv.first == "dtype")       f.dtype       = from_wire(kv.second.s);
                else if (kv.first == "is_primary")  f.is_primary  = kv.second.b;
                else if (kv.first == "auto_id")     f.auto_id     = kv.second.b;
                else if (kv.first == "max_length")  f.max_length  = static_cast<int>(kv.second.i);
                else if (kv.first == "dim")         f.dim         = static_cast<int>(kv.second.i);
                else if (kv.first == "description") f.description = kv.second.s;
            }
            fields.push_back(std::move(f));
        }
    }

    CollectionSchema schema(std::move(fields), opt_str("description"));
    auto metric = detail::metric_from_name(get_str("metric"));
    auto index  = detail::index_from_name(get_str("index"));

    uint64_t next_id = 1;
    auto nit = obj.find("next_id");
    if (nit != obj.end() && nit->second.tag == detail::JsonValue::Tag::Int)
        next_id = static_cast<uint64_t>(nit->second.i);

    std::map<uint64_t, Collection::Row> rows;
    auto rit = obj.find("rows");
    if (rit != obj.end() && rit->second.tag == detail::JsonValue::Tag::Object) {
        for (const auto& [k, v] : rit->second.obj) {
            if (v.tag != detail::JsonValue::Tag::Object) continue;
            uint64_t rid = std::stoull(k);
            Collection::Row r;
            for (const auto& [ck, cv] : v.obj)
                r[ck] = detail::json_to_value(cv);
            rows.emplace(rid, std::move(r));
        }
    }

    auto resolved_name = opt_str("name");
    if (resolved_name.empty()) resolved_name = name;

    auto db = std::make_unique<Database>(
        path,
        *schema.vector_field().dim,
        metric,
        index,
        opt.params ? &(*opt.params) : nullptr);

    return Collection::make_(
        std::move(resolved_name), std::move(path),
        std::move(schema), std::move(db),
        metric, index, std::move(rows),
        std::max<uint64_t>(next_id, 1));
}

} // namespace pistadb
