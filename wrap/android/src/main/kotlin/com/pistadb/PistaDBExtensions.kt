/*
 *    ___ _    _        ___  ___
 *   | _ (_)__| |_ __ _|   \| _ )
 *   |  _/ (_-<  _/ _` | |) | _ \
 *   |_| |_/__/\__\__,_|___/|___/
 */
@file:JvmName("PistaDBKt")

package com.pistadb

/**
 * Kotlin-idiomatic extensions and DSL builder for PistaDB.
 *
 * Usage:
 * ```kotlin
 * val db = pistaDB(filesDir.path + "/embeddings.pst", dim = 384) {
 *     metric    = Metric.COSINE
 *     indexType = IndexType.HNSW
 *     params { hnswEfSearch = 100 }
 * }
 *
 * db += VectorEntry(myEmbedding, label = "doc-42")
 * val hits = db.search(queryVec, k = 5)
 * hits.forEach { println("${it.label}  d=${it.distance}") }
 * ```
 */

/* ── DSL builder ─────────────────────────────────────────────────────────── */

/** Mutable configuration for the [pistaDB] DSL. */
class PistaDBConfig(val path: String, val dim: Int) {
    var metric: Metric       = Metric.L2
    var indexType: IndexType = IndexType.HNSW
    val params: PistaDBParams = PistaDBParams.defaults()

    /** Inline params configuration block. */
    fun params(block: PistaDBParams.() -> Unit) {
        params.block()
    }
}

/**
 * Opens or creates a PistaDB database using a configuration DSL.
 *
 * ```kotlin
 * val db = pistaDB(path, dim = 768) {
 *     metric    = Metric.COSINE
 *     indexType = IndexType.HNSW
 *     params    { hnswEfSearch = 50 }
 * }
 * ```
 */
fun pistaDB(path: String, dim: Int, block: PistaDBConfig.() -> Unit = {}): PistaDB {
    val cfg = PistaDBConfig(path, dim).apply(block)
    return PistaDB(cfg.path, cfg.dim, cfg.metric, cfg.indexType, cfg.params)
}

/* ── Operator overloads ──────────────────────────────────────────────────── */

/**
 * Inserts a vector using [VectorEntry] via the `+=` operator.
 *
 * ```kotlin
 * db += VectorEntry(floatArrayOf(...), "my-doc")
 * ```
 * @param entry Must carry a non-null [VectorEntry.vector]; requires that the
 *              caller supplies a unique id separately.
 */
operator fun PistaDB.plusAssign(pair: Pair<Long, VectorEntry>) {
    insert(pair.first, pair.second.vector, pair.second.label)
}

/** Retrieves a stored entry by id using the `[]` operator. */
operator fun PistaDB.get(id: Long): VectorEntry = get(id)

/* ── Batch helpers ───────────────────────────────────────────────────────── */

/**
 * Inserts an [Iterable] of (id, vector, label) triples.
 *
 * ```kotlin
 * db.insertAll(embeddings.mapIndexed { i, v -> Triple(i.toLong(), v, "doc-$i") })
 * ```
 */
fun PistaDB.insertAll(entries: Iterable<Triple<Long, FloatArray, String>>) {
    for ((id, vec, label) in entries) {
        insert(id, vec, label)
    }
}

/**
 * Returns search results as a [List] instead of an array.
 *
 * ```kotlin
 * val results: List<SearchResult> = db.searchList(queryVec, k = 10)
 * ```
 */
fun PistaDB.searchList(query: FloatArray, k: Int = 10): List<SearchResult> =
    search(query, k).toList()

/**
 * Returns only the ids from a search, sorted by ascending distance.
 *
 * ```kotlin
 * val nearestIds: List<Long> = db.searchIds(queryVec, k = 5)
 * ```
 */
fun PistaDB.searchIds(query: FloatArray, k: Int = 10): List<Long> =
    search(query, k).map { it.id }

/* ── use extension ───────────────────────────────────────────────────────── */

/**
 * Executes [block] with this database, saves, then closes – even if an
 * exception is thrown.
 *
 * ```kotlin
 * pistaDB(path, 384).use { db ->
 *     db.insert(1L, vec, "hello")
 * }  // saved and closed automatically
 * ```
 */
inline fun <T> PistaDB.use(block: (PistaDB) -> T): T {
    return try {
        block(this)
    } finally {
        close()
    }
}

/* ── PistaDBParams extensions ────────────────────────────────────────────── */

/** Fluent HNSW preset for high-recall scenarios. */
fun PistaDBParams.highRecallHnsw(): PistaDBParams = apply {
    hnswM              = 32
    hnswEfConstruction = 400
    hnswEfSearch       = 200
}

/** Fluent HNSW preset for low-latency scenarios. */
fun PistaDBParams.fastHnsw(): PistaDBParams = apply {
    hnswM              = 16
    hnswEfConstruction = 100
    hnswEfSearch       = 20
}
