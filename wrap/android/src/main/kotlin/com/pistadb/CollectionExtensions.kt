/*
 *    ___ _    _        ___  ___
 *   | _ (_)__| |_ __ _|   \| _ )
 *   |  _/ (_-<  _/ _` | |) | _ \
 *   |_| |_/__/\__\__,_|___/|___/
 */
@file:JvmName("CollectionKt")

package com.pistadb

/**
 * Kotlin-idiomatic helpers for [Collection].
 *
 * ```kotlin
 * val coll = collection(
 *     name        = "common_text",
 *     description = "Common text search",
 *     fields = listOf(
 *         field("lc_id",      DataType.INT64) { primary(true).autoId(true) },
 *         field("lc_section", DataType.VARCHAR)   { maxLength(100) },
 *         field("lc_vector",  DataType.FLOAT_VECTOR) { dim(1536) },
 *     ),
 * ) {
 *     metric  = Metric.COSINE
 *     index   = IndexType.HNSW
 *     baseDir = filesDir.absolutePath
 * }
 *
 * val ids = coll.insert(listOf(
 *     mapOf("lc_section" to "common", "lc_vector" to embedding),
 * ))
 * val hits = coll.search(query, k = 5)
 * coll.flush()
 * ```
 */

/** Builder receiver for [field]. */
class FieldBuilder internal constructor(name: String, dtype: DataType) {
    private val b = FieldSchema.Builder(name, dtype)
    fun primary(v: Boolean = true)     = also { b.primary(v) }
    fun autoId(v: Boolean = true)      = also { b.autoId(v) }
    fun maxLength(v: Int)              = also { b.maxLength(v) }
    fun dim(v: Int)                    = also { b.dim(v) }
    fun description(v: String)         = also { b.description(v) }
    internal fun build(): FieldSchema  = b.build()
}

/** Build a [FieldSchema] in a Kotlin DSL block. */
fun field(name: String, dtype: DataType, block: FieldBuilder.() -> Unit = {}): FieldSchema =
    FieldBuilder(name, dtype).apply(block).build()

/** Configuration for the [collection] DSL. */
class CollectionConfig {
    var metric: Metric         = Metric.L2
    var index:  IndexType      = IndexType.HNSW
    var params: PistaDBParams? = null
    var baseDir: String?       = null
    var path:    String?       = null
    var overwrite: Boolean     = false

    internal fun toOptions(): Collection.Options =
        Collection.Options()
            .metric(metric)
            .index(index)
            .params(params)
            .baseDir(baseDir)
            .path(path)
            .overwrite(overwrite)
}

/**
 * Create a new [Collection] with a Kotlin-idiomatic DSL.
 */
fun collection(
    name:        String,
    fields:      List<FieldSchema>,
    description: String = "",
    block:       CollectionConfig.() -> Unit = {},
): Collection {
    val cfg = CollectionConfig().apply(block)
    return Collection.create(name, fields, description, cfg.toOptions())
}

/**
 * Re-open a previously created collection.
 */
fun loadCollection(name: String, block: CollectionConfig.() -> Unit = {}): Collection {
    val cfg = CollectionConfig().apply(block)
    return Collection.load(name, cfg.toOptions())
}

/* ── Convenience extensions ──────────────────────────────────────────────── */

/** Insert a single row from a map. */
fun Collection.insert(row: Map<String, Any?>): Long =
    insert(listOf(row))[0]
