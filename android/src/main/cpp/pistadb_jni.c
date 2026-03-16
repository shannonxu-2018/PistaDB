/**
 *    ___ _    _        ___  ___
 *   | _ (_)__| |_ __ _|   \| _ )
 *   |  _/ (_-<  _/ _` | |) | _ \
 *   |_| |_/__/\__\__,_|___/|___/
 *
 * PistaDB Android JNI Bridge
 *
 * Maps Java native method declarations in com.pistadb.PistaDB to the
 * underlying C API.  The PistaDB* handle is stored on the Java side as a
 * long (jlong) and cast back here on every call.
 */
#include <jni.h>
#include <stdlib.h>
#include <string.h>
#include <android/log.h>
#include "pistadb.h"

#define LOG_TAG "PistaDB"
#define LOGE(...) __android_log_print(ANDROID_LOG_ERROR, LOG_TAG, __VA_ARGS__)

#define JNI_CLASS_SEARCH_RESULT "com/pistadb/SearchResult"
#define JNI_CLASS_VECTOR_ENTRY  "com/pistadb/VectorEntry"
#define JNI_CLASS_EXCEPTION     "com/pistadb/PistaDBException"

/* ── Helpers ─────────────────────────────────────────────────────────────── */

static void throw_exception(JNIEnv *env, const char *msg) {
    jclass cls = (*env)->FindClass(env, JNI_CLASS_EXCEPTION);
    if (cls) (*env)->ThrowNew(env, cls, msg);
    else LOGE("Could not find PistaDBException class");
}

static void throw_db_error(JNIEnv *env, PistaDB *db, const char *fallback) {
    const char *err = (db && pistadb_last_error(db)[0] != '\0')
                      ? pistadb_last_error(db) : fallback;
    throw_exception(env, err);
}

static inline PistaDB *handle_to_ptr(jlong handle) {
    return (PistaDB *)(intptr_t)handle;
}

/**
 * Extract PistaDBParams from a Java com.pistadb.PistaDBParams object.
 * Falls back to defaults when obj is NULL.
 */
static PistaDBParams extract_params(JNIEnv *env, jobject obj) {
    PistaDBParams p = pistadb_default_params();
    if (!obj) return p;

    jclass cls = (*env)->GetObjectClass(env, obj);
    if (!cls) return p;

#define GET_INT(c_field, java_field) do { \
    jfieldID fid = (*env)->GetFieldID(env, cls, java_field, "I"); \
    if (fid) p.c_field = (int)(*env)->GetIntField(env, obj, fid); \
} while (0)

#define GET_FLOAT(c_field, java_field) do { \
    jfieldID fid = (*env)->GetFieldID(env, cls, java_field, "F"); \
    if (fid) p.c_field = (float)(*env)->GetFloatField(env, obj, fid); \
} while (0)

    GET_INT(hnsw_M,               "hnswM");
    GET_INT(hnsw_ef_construction, "hnswEfConstruction");
    GET_INT(hnsw_ef_search,       "hnswEfSearch");
    GET_INT(ivf_nlist,            "ivfNlist");
    GET_INT(ivf_nprobe,           "ivfNprobe");
    GET_INT(pq_M,                 "pqM");
    GET_INT(pq_nbits,             "pqNbits");
    GET_INT(diskann_R,            "diskannR");
    GET_INT(diskann_L,            "diskannL");
    GET_FLOAT(diskann_alpha,      "diskannAlpha");
    GET_INT(lsh_L,                "lshL");
    GET_INT(lsh_K,                "lshK");
    GET_FLOAT(lsh_w,              "lshW");
    GET_INT(scann_nlist,          "scannNlist");
    GET_INT(scann_nprobe,         "scannNprobe");
    GET_INT(scann_pq_M,           "scannPqM");
    GET_INT(scann_pq_bits,        "scannPqBits");
    GET_INT(scann_rerank_k,       "scannRerankK");
    GET_FLOAT(scann_aq_eta,       "scannAqEta");

#undef GET_INT
#undef GET_FLOAT

    return p;
}

/* ── Lifecycle ───────────────────────────────────────────────────────────── */

JNIEXPORT jlong JNICALL
Java_com_pistadb_PistaDB_nativeOpen(JNIEnv *env, jclass klass,
                                    jstring j_path, jint dim,
                                    jint metric, jint index_type,
                                    jobject j_params)
{
    const char *path = (*env)->GetStringUTFChars(env, j_path, NULL);
    if (!path) return 0L;

    PistaDBParams params = extract_params(env, j_params);

    PistaDB *db = pistadb_open(path, (int)dim,
                               (PistaDBMetric)metric,
                               (PistaDBIndexType)index_type,
                               &params);
    (*env)->ReleaseStringUTFChars(env, j_path, path);

    if (!db) {
        throw_exception(env, "pistadb_open failed – check path and parameters");
        return 0L;
    }
    return (jlong)(intptr_t)db;
}

JNIEXPORT void JNICALL
Java_com_pistadb_PistaDB_nativeClose(JNIEnv *env, jclass klass, jlong handle)
{
    PistaDB *db = handle_to_ptr(handle);
    if (db) pistadb_close(db);
}

JNIEXPORT void JNICALL
Java_com_pistadb_PistaDB_nativeSave(JNIEnv *env, jclass klass, jlong handle)
{
    PistaDB *db = handle_to_ptr(handle);
    if (!db) { throw_exception(env, "Null database handle"); return; }
    if (pistadb_save(db) != PISTADB_OK)
        throw_db_error(env, db, "pistadb_save failed");
}

/* ── CRUD ────────────────────────────────────────────────────────────────── */

JNIEXPORT void JNICALL
Java_com_pistadb_PistaDB_nativeInsert(JNIEnv *env, jclass klass, jlong handle,
                                      jlong id, jstring j_label,
                                      jfloatArray j_vec)
{
    PistaDB *db = handle_to_ptr(handle);
    if (!db) { throw_exception(env, "Null database handle"); return; }

    const char *label = j_label
                        ? (*env)->GetStringUTFChars(env, j_label, NULL)
                        : NULL;
    jfloat *vec = (*env)->GetFloatArrayElements(env, j_vec, NULL);
    if (!vec) {
        if (label) (*env)->ReleaseStringUTFChars(env, j_label, label);
        throw_exception(env, "Failed to pin float array");
        return;
    }

    int rc = pistadb_insert(db, (uint64_t)id, label, (const float *)vec);

    (*env)->ReleaseFloatArrayElements(env, j_vec, vec, JNI_ABORT);
    if (label) (*env)->ReleaseStringUTFChars(env, j_label, label);

    if (rc != PISTADB_OK) throw_db_error(env, db, "pistadb_insert failed");
}

JNIEXPORT void JNICALL
Java_com_pistadb_PistaDB_nativeDelete(JNIEnv *env, jclass klass,
                                      jlong handle, jlong id)
{
    PistaDB *db = handle_to_ptr(handle);
    if (!db) { throw_exception(env, "Null database handle"); return; }
    if (pistadb_delete(db, (uint64_t)id) != PISTADB_OK)
        throw_db_error(env, db, "pistadb_delete failed");
}

JNIEXPORT void JNICALL
Java_com_pistadb_PistaDB_nativeUpdate(JNIEnv *env, jclass klass,
                                      jlong handle, jlong id,
                                      jfloatArray j_vec)
{
    PistaDB *db = handle_to_ptr(handle);
    if (!db) { throw_exception(env, "Null database handle"); return; }

    jfloat *vec = (*env)->GetFloatArrayElements(env, j_vec, NULL);
    if (!vec) { throw_exception(env, "Failed to pin float array"); return; }

    int rc = pistadb_update(db, (uint64_t)id, (const float *)vec);
    (*env)->ReleaseFloatArrayElements(env, j_vec, vec, JNI_ABORT);

    if (rc != PISTADB_OK) throw_db_error(env, db, "pistadb_update failed");
}

JNIEXPORT jobject JNICALL
Java_com_pistadb_PistaDB_nativeGet(JNIEnv *env, jclass klass,
                                   jlong handle, jlong id)
{
    PistaDB *db = handle_to_ptr(handle);
    if (!db) { throw_exception(env, "Null database handle"); return NULL; }

    int dim = pistadb_dim(db);
    float *vec = (float *)malloc((size_t)dim * sizeof(float));
    if (!vec) { throw_exception(env, "Out of memory"); return NULL; }

    char label[256] = {0};
    int rc = pistadb_get(db, (uint64_t)id, vec, label);
    if (rc != PISTADB_OK) {
        free(vec);
        throw_db_error(env, db, "pistadb_get failed");
        return NULL;
    }

    jclass entry_cls = (*env)->FindClass(env, JNI_CLASS_VECTOR_ENTRY);
    if (!entry_cls) { free(vec); return NULL; }

    jmethodID ctor = (*env)->GetMethodID(env, entry_cls,
                                         "<init>", "([FLjava/lang/String;)V");
    if (!ctor) { free(vec); return NULL; }

    jfloatArray j_vec = (*env)->NewFloatArray(env, dim);
    if (!j_vec) { free(vec); return NULL; }
    (*env)->SetFloatArrayRegion(env, j_vec, 0, dim, vec);
    free(vec);

    jstring j_label = (*env)->NewStringUTF(env, label);
    return (*env)->NewObject(env, entry_cls, ctor, j_vec, j_label);
}

/* ── Search ──────────────────────────────────────────────────────────────── */

JNIEXPORT jobjectArray JNICALL
Java_com_pistadb_PistaDB_nativeSearch(JNIEnv *env, jclass klass,
                                      jlong handle, jfloatArray j_query,
                                      jint k)
{
    PistaDB *db = handle_to_ptr(handle);
    if (!db) { throw_exception(env, "Null database handle"); return NULL; }

    jfloat *query = (*env)->GetFloatArrayElements(env, j_query, NULL);
    if (!query) { throw_exception(env, "Failed to pin query array"); return NULL; }

    PistaDBResult *results = (PistaDBResult *)malloc((size_t)k * sizeof(PistaDBResult));
    if (!results) {
        (*env)->ReleaseFloatArrayElements(env, j_query, query, JNI_ABORT);
        throw_exception(env, "Out of memory");
        return NULL;
    }

    int n = pistadb_search(db, (const float *)query, (int)k, results);
    (*env)->ReleaseFloatArrayElements(env, j_query, query, JNI_ABORT);

    if (n < 0) {
        free(results);
        throw_db_error(env, db, "pistadb_search failed");
        return NULL;
    }

    jclass sr_cls = (*env)->FindClass(env, JNI_CLASS_SEARCH_RESULT);
    if (!sr_cls) { free(results); return NULL; }

    /* (JFLjava/lang/String;)V  ←  SearchResult(long id, float dist, String label) */
    jmethodID ctor = (*env)->GetMethodID(env, sr_cls,
                                         "<init>", "(JFLjava/lang/String;)V");
    if (!ctor) { free(results); return NULL; }

    jobjectArray arr = (*env)->NewObjectArray(env, n, sr_cls, NULL);
    if (!arr) { free(results); return NULL; }

    for (int i = 0; i < n; i++) {
        jstring j_label = (*env)->NewStringUTF(env, results[i].label);
        jobject sr = (*env)->NewObject(env, sr_cls, ctor,
                                       (jlong)results[i].id,
                                       (jfloat)results[i].distance,
                                       j_label);
        (*env)->SetObjectArrayElement(env, arr, i, sr);
        (*env)->DeleteLocalRef(env, sr);
        (*env)->DeleteLocalRef(env, j_label);
    }

    free(results);
    return arr;
}

/* ── Index management ────────────────────────────────────────────────────── */

JNIEXPORT void JNICALL
Java_com_pistadb_PistaDB_nativeTrain(JNIEnv *env, jclass klass, jlong handle)
{
    PistaDB *db = handle_to_ptr(handle);
    if (!db) { throw_exception(env, "Null database handle"); return; }
    if (pistadb_train(db) != PISTADB_OK)
        throw_db_error(env, db, "pistadb_train failed");
}

/* ── Metadata ────────────────────────────────────────────────────────────── */

JNIEXPORT jint JNICALL
Java_com_pistadb_PistaDB_nativeCount(JNIEnv *env, jclass klass, jlong handle)
{
    PistaDB *db = handle_to_ptr(handle);
    return db ? (jint)pistadb_count(db) : 0;
}

JNIEXPORT jint JNICALL
Java_com_pistadb_PistaDB_nativeDim(JNIEnv *env, jclass klass, jlong handle)
{
    PistaDB *db = handle_to_ptr(handle);
    return db ? (jint)pistadb_dim(db) : 0;
}

JNIEXPORT jint JNICALL
Java_com_pistadb_PistaDB_nativeMetric(JNIEnv *env, jclass klass, jlong handle)
{
    PistaDB *db = handle_to_ptr(handle);
    return db ? (jint)pistadb_metric(db) : 0;
}

JNIEXPORT jint JNICALL
Java_com_pistadb_PistaDB_nativeIndexType(JNIEnv *env, jclass klass, jlong handle)
{
    PistaDB *db = handle_to_ptr(handle);
    return db ? (jint)pistadb_index_type(db) : 0;
}

JNIEXPORT jstring JNICALL
Java_com_pistadb_PistaDB_nativeLastError(JNIEnv *env, jclass klass, jlong handle)
{
    PistaDB *db = handle_to_ptr(handle);
    return (*env)->NewStringUTF(env, db ? pistadb_last_error(db) : "null handle");
}

JNIEXPORT jstring JNICALL
Java_com_pistadb_PistaDB_nativeVersion(JNIEnv *env, jclass klass)
{
    return (*env)->NewStringUTF(env, pistadb_version());
}
