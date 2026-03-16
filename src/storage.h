/**
 *    ___ _    _        ___  ___
 *   | _ (_)__| |_ __ _|   \| _ )
 *   |  _/ (_-<  _/ _` | |) | _ \
 *   |_| |_/__/\__\__,_|___/|___/
 *
 * PistaDB - storage.h
 * Single-file binary storage format.
 *
 * ┌──────────────────────────────────┐
 * │  File Header  (128 bytes fixed)  │
 * ├──────────────────────────────────┤
 * │  Vector Data Section             │
 * │  (at offset header.vec_offset)   │
 * ├──────────────────────────────────┤
 * │  Index Section                   │
 * │  (at offset header.idx_offset)   │
 * └──────────────────────────────────┘
 *
 * The header contains absolute file offsets so both sections can be
 * positioned independently and the format stays forward-compatible:
 * a newer reader that finds an unknown section simply skips it.
 *
 * Backward compatibility:
 *   - version_major change = breaking (refuse to open)
 *   - version_minor change = additive only (older reader OK)
 *   - reserved bytes MUST be zero on write
 */
#ifndef PISTADB_STORAGE_H
#define PISTADB_STORAGE_H

#include "pistadb_types.h"
#include <stdint.h>
#include <stdio.h>

/* ── On-disk file header (exactly 128 bytes) ─────────────────────────────── */
#pragma pack(push, 1)
typedef struct {
    char     magic[4];           /* "PSDB"                          */
    uint16_t version_major;      /* breaking changes                */
    uint16_t version_minor;      /* additive changes                */
    uint32_t flags;              /* feature flags (reserved, = 0)   */
    uint32_t dimension;          /* vector dimension                */
    uint16_t metric_type;        /* PistaDBMetric                    */
    uint16_t index_type;         /* PistaDBIndexType                 */
    uint64_t num_vectors;        /* total (including deleted)       */
    uint64_t next_id;            /* auto-increment counter          */
    uint64_t vec_offset;         /* offset of Vector Data Section   */
    uint64_t vec_size;           /* byte size of Vector Data Section*/
    uint64_t idx_offset;         /* offset of Index Section         */
    uint64_t idx_size;           /* byte size of Index Section      */
    uint8_t  reserved[56];       /* pad to 128 bytes; must be zero  */
    uint32_t header_crc;         /* CRC32 of bytes [0..123]         */
} PistaDBFileHeader;
#pragma pack(pop)

/* Compile-time size check (C11 _Static_assert; falls back to enum trick) */
#if defined(__STDC_VERSION__) && __STDC_VERSION__ >= 201112L
_Static_assert(sizeof(PistaDBFileHeader) == 128, "PistaDBFileHeader must be 128 bytes");
#elif defined(__cplusplus)
static_assert(sizeof(PistaDBFileHeader) == 128, "PistaDBFileHeader must be 128 bytes");
#endif

/* ── API ─────────────────────────────────────────────────────────────────── */

/**
 * Write the complete database to a file.
 * @param vec_buf   serialised vector section (from index's own serialiser)
 * @param idx_buf   serialised index section
 */
int storage_write(const char *path,
                  PistaDBMetric metric, PistaDBIndexType idx_type,
                  uint32_t dim, uint64_t num_vectors, uint64_t next_id,
                  const void *vec_buf, size_t vec_size,
                  const void *idx_buf, size_t idx_size);

/**
 * Read header from file.  Validates magic, version, and CRC.
 * @param hdr  output header struct
 */
int storage_read_header(const char *path, PistaDBFileHeader *hdr);

/**
 * Read the raw vector and index blobs from an open file.
 * Caller is responsible for freeing *vec_buf and *idx_buf.
 */
int storage_read_sections(const char *path, const PistaDBFileHeader *hdr,
                          void **vec_buf, size_t *vec_size,
                          void **idx_buf, size_t *idx_size);

#endif /* PISTADB_STORAGE_H */
