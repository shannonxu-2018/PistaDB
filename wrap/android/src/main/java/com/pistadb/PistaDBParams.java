/*
 *    ___ _    _        ___  ___
 *   | _ (_)__| |_ __ _|   \| _ )
 *   |  _/ (_-<  _/ _` | |) | _ \
 *   |_| |_/__/\__\__,_|___/|___/
 */
package com.pistadb;

/**
 * Index tuning parameters passed when opening or creating a database.
 *
 * <p>Defaults match the C {@code pistadb_default_params()} values.
 * Use the {@link Builder} for a fluent, self-documenting configuration:
 *
 * <pre>{@code
 * PistaDBParams params = PistaDBParams.builder()
 *         .hnswM(32)
 *         .hnswEfSearch(100)
 *         .build();
 * }</pre>
 *
 * <p>Field names intentionally mirror the C struct (camelCase) so they can be
 * read by JNI via reflection without a mapping table.
 */
public class PistaDBParams {

    /* HNSW ----------------------------------------------------------------- */

    /** Max connections per layer (default 16). */
    public int hnswM               = 16;
    /** Build-time search width (default 200). Higher → better recall, slower build. */
    public int hnswEfConstruction  = 200;
    /** Query-time search width (default 50). Higher → better recall, slower query. */
    public int hnswEfSearch        = 50;

    /* IVF / IVF_PQ --------------------------------------------------------- */

    /** Number of centroids (default 128). */
    public int ivfNlist            = 128;
    /** Centroids to probe at search time (default 8). */
    public int ivfNprobe           = 8;
    /** PQ subspaces (default 8; must divide dim evenly). */
    public int pqM                 = 8;
    /** Bits per sub-code: 4 or 8 (default 8). */
    public int pqNbits             = 8;

    /* DiskANN -------------------------------------------------------------- */

    /** Max graph degree (default 32). */
    public int   diskannR          = 32;
    /** Build search list size (default 100). */
    public int   diskannL          = 100;
    /** Pruning parameter α (default 1.2f). */
    public float diskannAlpha      = 1.2f;

    /* LSH ------------------------------------------------------------------ */

    /** Number of hash tables (default 10). */
    public int   lshL              = 10;
    /** Hash functions per table (default 8). */
    public int   lshK              = 8;
    /** Bucket width for E2LSH (default 10.0f). */
    public float lshW              = 10.0f;

    /* ScaNN ---------------------------------------------------------------- */

    /** Coarse IVF partitions (default 128). */
    public int   scannNlist        = 128;
    /** Partitions to probe at search time (default 32). */
    public int   scannNprobe       = 32;
    /** PQ sub-spaces (default 8). */
    public int   scannPqM          = 8;
    /** Bits per sub-code: 4 or 8 (default 8). */
    public int   scannPqBits       = 8;
    /** Candidates to exact-rerank (default 100). */
    public int   scannRerankK      = 100;
    /** Anisotropic penalty η (default 0.2f). */
    public float scannAqEta        = 0.2f;

    /** Returns a params instance with all defaults applied. */
    public static PistaDBParams defaults() {
        return new PistaDBParams();
    }

    /** Returns a {@link Builder} seeded with default values. */
    public static Builder builder() {
        return new Builder();
    }

    /* ── Builder ─────────────────────────────────────────────────────────── */

    public static final class Builder {
        private final PistaDBParams p = new PistaDBParams();

        public Builder hnswM(int v)              { p.hnswM = v;              return this; }
        public Builder hnswEfConstruction(int v) { p.hnswEfConstruction = v; return this; }
        public Builder hnswEfSearch(int v)       { p.hnswEfSearch = v;       return this; }

        public Builder ivfNlist(int v)           { p.ivfNlist = v;           return this; }
        public Builder ivfNprobe(int v)          { p.ivfNprobe = v;          return this; }
        public Builder pqM(int v)                { p.pqM = v;                return this; }
        public Builder pqNbits(int v)            { p.pqNbits = v;            return this; }

        public Builder diskannR(int v)           { p.diskannR = v;           return this; }
        public Builder diskannL(int v)           { p.diskannL = v;           return this; }
        public Builder diskannAlpha(float v)     { p.diskannAlpha = v;       return this; }

        public Builder lshL(int v)               { p.lshL = v;               return this; }
        public Builder lshK(int v)               { p.lshK = v;               return this; }
        public Builder lshW(float v)             { p.lshW = v;               return this; }

        public Builder scannNlist(int v)         { p.scannNlist = v;         return this; }
        public Builder scannNprobe(int v)        { p.scannNprobe = v;        return this; }
        public Builder scannPqM(int v)           { p.scannPqM = v;           return this; }
        public Builder scannPqBits(int v)        { p.scannPqBits = v;        return this; }
        public Builder scannRerankK(int v)       { p.scannRerankK = v;       return this; }
        public Builder scannAqEta(float v)       { p.scannAqEta = v;         return this; }

        public PistaDBParams build() {
            return p;
        }
    }
}
