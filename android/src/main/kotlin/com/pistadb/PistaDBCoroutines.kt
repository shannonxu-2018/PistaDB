/*
 *    ___ _    _        ___  ___
 *   | _ (_)__| |_ __ _|   \| _ )
 *   |  _/ (_-<  _/ _` | |) | _ \
 *   |_| |_/__/\__\__,_|___/|___/
 */
package com.pistadb

import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.withContext

/**
 * Coroutine-friendly suspend wrappers for PistaDB.
 *
 * All functions dispatch blocking native work to [Dispatchers.IO] so they
 * are safe to call from the main thread or any coroutine scope.
 *
 * ```kotlin
 * lifecycleScope.launch {
 *     db.insertAsync(id = 1L, vector = embedding, label = "doc")
 *     val results = db.searchAsync(queryVec, k = 5)
 *     results.forEach { Log.d("TAG", it.label) }
 * }
 * ```
 */

/** Suspend wrapper for [PistaDB.insert]. */
suspend fun PistaDB.insertAsync(
    id: Long,
    vector: FloatArray,
    label: String = ""
): Unit = withContext(Dispatchers.IO) {
    insert(id, vector, label)
}

/** Suspend wrapper for [PistaDB.delete]. */
suspend fun PistaDB.deleteAsync(id: Long): Unit = withContext(Dispatchers.IO) {
    delete(id)
}

/** Suspend wrapper for [PistaDB.update]. */
suspend fun PistaDB.updateAsync(id: Long, vector: FloatArray): Unit =
    withContext(Dispatchers.IO) {
        update(id, vector)
    }

/** Suspend wrapper for [PistaDB.get]. */
suspend fun PistaDB.getAsync(id: Long): VectorEntry = withContext(Dispatchers.IO) {
    get(id)
}

/**
 * Suspend wrapper for [PistaDB.search].
 *
 * @return List of results ordered by ascending distance.
 */
suspend fun PistaDB.searchAsync(
    query: FloatArray,
    k: Int = 10
): List<SearchResult> = withContext(Dispatchers.IO) {
    search(query, k).toList()
}

/** Suspend wrapper for [PistaDB.train]. */
suspend fun PistaDB.trainAsync(): Unit = withContext(Dispatchers.IO) {
    train()
}

/** Suspend wrapper for [PistaDB.save]. */
suspend fun PistaDB.saveAsync(): Unit = withContext(Dispatchers.IO) {
    save()
}

/**
 * Suspend batch insert.
 *
 * ```kotlin
 * db.insertBatchAsync(
 *     entries = embeddings.mapIndexed { i, v -> Triple(i.toLong(), v, "doc-$i") }
 * )
 * ```
 */
suspend fun PistaDB.insertBatchAsync(
    entries: List<Triple<Long, FloatArray, String>>
): Unit = withContext(Dispatchers.IO) {
    for ((id, vec, label) in entries) {
        insert(id, vec, label)
    }
}
