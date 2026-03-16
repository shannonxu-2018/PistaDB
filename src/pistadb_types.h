/**
 *    ___ _    _        ___  ___
 *   | _ (_)__| |_ __ _|   \| _ )
 *   |  _/ (_-<  _/ _` | |) | _ \
 *   |_| |_/__/\__\__,_|___/|___/
 *
 * PistaDB - Lightweight Embedded Vector Database
 * pistadb_types.h - Common types and error codes
 *
 * File format version: 1.0
 * Compatible down to: 1.0
 */
#ifndef PISTADB_TYPES_H
#define PISTADB_TYPES_H

#include <stdint.h>
#include <stddef.h>

/* ── Error codes ─────────────────────────────────────────────────────────── */
#define PISTADB_OK           0
#define PISTADB_ERR         -1   /* generic error              */
#define PISTADB_ENOMEM      -2   /* allocation failure         */
#define PISTADB_EIO         -3   /* I/O error                  */
#define PISTADB_ENOTFOUND   -4   /* vector not found           */
#define PISTADB_EEXIST      -5   /* vector id already exists   */
#define PISTADB_EINVAL      -6   /* invalid argument           */
#define PISTADB_ENOTRAINED  -7   /* index not trained          */
#define PISTADB_ECORRUPT    -8   /* corrupted file             */
#define PISTADB_EVERSION    -9   /* incompatible file version  */

/* ── Distance metric types ───────────────────────────────────────────────── */
typedef enum {
    METRIC_L2      = 0,   /* Euclidean distance                   */
    METRIC_COSINE  = 1,   /* Cosine similarity (as distance)      */
    METRIC_IP      = 2,   /* Inner product (negative, as distance)*/
    METRIC_L1      = 3,   /* Manhattan distance                   */
    METRIC_HAMMING = 4    /* Hamming distance (bit/element count) */
} PistaDBMetric;

/* ── Index algorithm types ───────────────────────────────────────────────── */
typedef enum {
    INDEX_LINEAR  = 0,   /* Brute-force linear scan                  */
    INDEX_HNSW    = 1,   /* Hierarchical NSW                         */
    INDEX_IVF     = 2,   /* Inverted File Index                      */
    INDEX_IVF_PQ  = 3,   /* IVF + Product Quantization               */
    INDEX_DISKANN = 4,   /* Vamana / DiskANN                         */
    INDEX_LSH     = 5,   /* Locality-Sensitive Hashing               */
    INDEX_SCANN   = 6    /* ScaNN: Anisotropic Vector Quantization   */
} PistaDBIndexType;

/* ── Single search result ────────────────────────────────────────────────── */
typedef struct {
    uint64_t  id;
    float     distance;
    char      label[256];
} PistaDBResult;

/* ── Index-specific parameters (passed at open/create time) ─────────────── */
typedef struct {
    /* HNSW */
    int   hnsw_M;                /* max connections per layer (default 16)    */
    int   hnsw_ef_construction;  /* build-time search width   (default 200)   */
    int   hnsw_ef_search;        /* query-time search width   (default 50)    */

    /* IVF / IVF_PQ */
    int   ivf_nlist;             /* number of centroids       (default 128)   */
    int   ivf_nprobe;            /* centroids to search       (default 8)     */
    int   pq_M;                  /* PQ subspaces              (default 8)     */
    int   pq_nbits;              /* bits per sub-code (4 or 8, default 8)     */

    /* DiskANN / Vamana */
    int   diskann_R;             /* max graph degree          (default 32)    */
    int   diskann_L;             /* build search list size    (default 100)   */
    float diskann_alpha;         /* pruning parameter         (default 1.2)   */

    /* LSH */
    int   lsh_L;                 /* number of hash tables     (default 10)    */
    int   lsh_K;                 /* hash functions per table  (default 8)     */
    float lsh_w;                 /* bucket width (E2LSH)      (default 4.0)   */

    /* ScaNN (Anisotropic Vector Quantization) */
    int   scann_nlist;           /* coarse IVF partitions     (default 128)   */
    int   scann_nprobe;          /* partitions to probe       (default 32)    */
    int   scann_pq_M;            /* PQ sub-spaces             (default 8)     */
    int   scann_pq_bits;         /* bits per sub-code (4|8)   (default 8)     */
    int   scann_rerank_k;        /* candidates to rerank      (default 100)   */
    float scann_aq_eta;          /* anisotropic penalty η     (default 0.2)   */
} PistaDBParams;

/* Default parameter initialiser */
static inline PistaDBParams pistadb_default_params(void) {
    PistaDBParams p;
    p.hnsw_M               = 16;
    p.hnsw_ef_construction = 200;
    p.hnsw_ef_search       = 50;
    p.ivf_nlist            = 128;
    p.ivf_nprobe           = 8;
    p.pq_M                 = 8;
    p.pq_nbits             = 8;
    p.diskann_R            = 32;
    p.diskann_L            = 100;
    p.diskann_alpha        = 1.2f;
    p.lsh_L                = 10;
    p.lsh_K                = 8;
    p.lsh_w                = 10.0f;
    p.scann_nlist          = 128;
    p.scann_nprobe         = 32;
    p.scann_pq_M           = 8;
    p.scann_pq_bits        = 8;
    p.scann_rerank_k       = 100;
    p.scann_aq_eta         = 0.2f;
    return p;
}

/* ── File format magic & version ─────────────────────────────────────────── */
#define PISTADB_MAGIC         "PSDB"
#define PISTADB_MAGIC_LEN     4
#define PISTADB_VERSION_MAJOR 1
#define PISTADB_VERSION_MINOR 0

#endif /* PISTADB_TYPES_H */
