/**
 *    ___ _    _        ___  ___
 *   | _ (_)__| |_ __ _|   \| _ )
 *   |  _/ (_-<  _/ _` | |) | _ \
 *   |_| |_/__/\__\__,_|___/|___/
 *
 * PistaDB - storage.c
 */
#ifndef _WIN32
#  define _FILE_OFFSET_BITS 64   /* enable 64-bit fseeko/off_t on 32-bit POSIX */
#endif
#include "storage.h"
#include "utils.h"
#include <stdlib.h>
#include <string.h>
#include <stdio.h>

/* 64-bit fseek: fseek(long) is 32-bit on Windows, silently truncates >2GB offsets. */
#ifdef _WIN32
#  define FSEEK64(f, off, whence) _fseeki64((f), (__int64)(off), (whence))
#else
#  define FSEEK64(f, off, whence) fseeko((f), (off_t)(off), (whence))
#endif

int storage_write(const char *path,
                  PistaDBMetric metric, PistaDBIndexType idx_type,
                  uint32_t dim, uint64_t num_vectors, uint64_t next_id,
                  const void *vec_buf, size_t vec_size,
                  const void *idx_buf, size_t idx_size) {
    FILE *f = fopen(path, "wb");
    if (!f) return PISTADB_EIO;

    PistaDBFileHeader hdr;
    memset(&hdr, 0, sizeof(hdr));
    memcpy(hdr.magic, PISTADB_MAGIC, 4);
    hdr.version_major = PISTADB_VERSION_MAJOR;
    hdr.version_minor = PISTADB_VERSION_MINOR;
    hdr.flags         = 0;
    hdr.dimension     = dim;
    hdr.metric_type   = (uint16_t)metric;
    hdr.index_type    = (uint16_t)idx_type;
    hdr.num_vectors   = num_vectors;
    hdr.next_id       = next_id;
    hdr.vec_offset    = sizeof(PistaDBFileHeader);
    hdr.vec_size      = (uint64_t)vec_size;
    hdr.idx_offset    = hdr.vec_offset + (uint64_t)vec_size;
    hdr.idx_size      = (uint64_t)idx_size;
    hdr.header_crc    = crc32_compute(&hdr, 124);  /* CRC of bytes 0..123 */

    int ok = 1;
    ok = ok && (fwrite(&hdr,    1, sizeof(hdr), f) == sizeof(hdr));
    ok = ok && (fwrite(vec_buf, 1, vec_size,    f) == vec_size);
    ok = ok && (fwrite(idx_buf, 1, idx_size,    f) == idx_size);
    fclose(f);
    return ok ? PISTADB_OK : PISTADB_EIO;
}

int storage_read_header(const char *path, PistaDBFileHeader *hdr) {
    FILE *f = fopen(path, "rb");
    if (!f) return PISTADB_EIO;

    if (fread(hdr, 1, sizeof(*hdr), f) != sizeof(*hdr)) {
        fclose(f); return PISTADB_EIO;
    }
    fclose(f);

    /* Magic */
    if (memcmp(hdr->magic, PISTADB_MAGIC, 4) != 0) return PISTADB_ECORRUPT;

    /* Version check */
    if (hdr->version_major != PISTADB_VERSION_MAJOR) return PISTADB_EVERSION;

    /* CRC */
    uint32_t crc = crc32_compute(hdr, 124);
    if (crc != hdr->header_crc) return PISTADB_ECORRUPT;

    return PISTADB_OK;
}

int storage_read_sections(const char *path, const PistaDBFileHeader *hdr,
                          void **vec_buf, size_t *vec_size,
                          void **idx_buf, size_t *idx_size) {
    FILE *f = fopen(path, "rb");
    if (!f) return PISTADB_EIO;

    *vec_size = (size_t)hdr->vec_size;
    *idx_size = (size_t)hdr->idx_size;
    *vec_buf  = malloc(*vec_size + 1);
    *idx_buf  = malloc(*idx_size + 1);
    if (!*vec_buf || !*idx_buf) {
        free(*vec_buf); free(*idx_buf);
        fclose(f); return PISTADB_ENOMEM;
    }

    int ok = 1;
    ok = ok && (FSEEK64(f, hdr->vec_offset, SEEK_SET) == 0);
    ok = ok && (fread(*vec_buf, 1, *vec_size, f) == *vec_size);
    ok = ok && (FSEEK64(f, hdr->idx_offset, SEEK_SET) == 0);
    ok = ok && (fread(*idx_buf, 1, *idx_size, f) == *idx_size);
    fclose(f);

    if (!ok) {
        free(*vec_buf); free(*idx_buf);
        *vec_buf = *idx_buf = NULL;
        return PISTADB_EIO;
    }
    return PISTADB_OK;
}
