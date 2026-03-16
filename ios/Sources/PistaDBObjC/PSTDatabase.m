/*
 *    ___ _    _        ___  ___
 *   | _ (_)__| |_ __ _|   \| _ )
 *   |  _/ (_-<  _/ _` | |) | _ \
 *   |_| |_/__/\__\__,_|___/|___/
 */
#import "PSTDatabase.h"

/* ── C header resolution ──────────────────────────────────────────────────── */
/* Works under both SPM (module name CPistaDB) and a manual Xcode project     */
/* with the src/ directory on the header search path.                          */
#if __has_include(<CPistaDB/pistadb.h>)
#  include <CPistaDB/pistadb.h>
#elif __has_include("pistadb.h")
#  include "pistadb.h"
#else
#  error "Cannot find pistadb.h – add the PistaDB src/ directory to Header Search Paths"
#endif

/* ── Error domain ─────────────────────────────────────────────────────────── */

NSErrorDomain const PSTDatabaseErrorDomain = @"com.pistadb.error";

/* ── Private helpers ──────────────────────────────────────────────────────── */

@implementation PSTSearchResult

- (instancetype)initWithId:(uint64_t)vectorId distance:(float)distance label:(NSString *)label {
    if (!(self = [super init])) return nil;
    _vectorId = vectorId;
    _distance = distance;
    _label    = label ?: @"";
    return self;
}

- (NSString *)description {
    return [NSString stringWithFormat:@"<PSTSearchResult id=%llu dist=%.6f label=%@>",
            _vectorId, _distance, _label];
}

@end

/* ─────────────────────────────────────────────────────────────────────────── */

@implementation PSTVectorEntry {
    NSData *_data;
}

- (instancetype)initWithFloats:(const float *)floats count:(NSInteger)count label:(NSString *)label {
    if (!(self = [super init])) return nil;
    _data  = [NSData dataWithBytes:floats length:(NSUInteger)(count * sizeof(float))];
    _label = label ?: @"";
    return self;
}

- (NSData *)vectorData { return _data; }
- (NSInteger)count     { return (NSInteger)(_data.length / sizeof(float)); }

- (float)floatAtIndex:(NSInteger)index {
    const float *p = (const float *)_data.bytes;
    return p[index];
}

- (NSString *)description {
    return [NSString stringWithFormat:@"<PSTVectorEntry dim=%ld label=%@>",
            (long)self.count, _label];
}

@end

/* ── PSTParams ────────────────────────────────────────────────────────────── */

@implementation PSTParams

+ (instancetype)defaultParams {
    return [[self alloc] init];
}

- (instancetype)init {
    if (!(self = [super init])) return nil;
    /* Mirror pistadb_default_params() */
    _hnswM               = 16;
    _hnswEfConstruction  = 200;
    _hnswEfSearch        = 50;
    _ivfNlist            = 128;
    _ivfNprobe           = 8;
    _pqM                 = 8;
    _pqNbits             = 8;
    _diskannR            = 32;
    _diskannL            = 100;
    _diskannAlpha        = 1.2f;
    _lshL                = 10;
    _lshK                = 8;
    _lshW                = 10.0f;
    _scannNlist          = 128;
    _scannNprobe         = 32;
    _scannPqM            = 8;
    _scannPqBits         = 8;
    _scannRerankK        = 100;
    _scannAqEta          = 0.2f;
    return self;
}

- (id)copyWithZone:(NSZone *)zone {
    PSTParams *copy = [[PSTParams allocWithZone:zone] init];
    copy.hnswM               = _hnswM;
    copy.hnswEfConstruction  = _hnswEfConstruction;
    copy.hnswEfSearch        = _hnswEfSearch;
    copy.ivfNlist            = _ivfNlist;
    copy.ivfNprobe           = _ivfNprobe;
    copy.pqM                 = _pqM;
    copy.pqNbits             = _pqNbits;
    copy.diskannR            = _diskannR;
    copy.diskannL            = _diskannL;
    copy.diskannAlpha        = _diskannAlpha;
    copy.lshL                = _lshL;
    copy.lshK                = _lshK;
    copy.lshW                = _lshW;
    copy.scannNlist          = _scannNlist;
    copy.scannNprobe         = _scannNprobe;
    copy.scannPqM            = _scannPqM;
    copy.scannPqBits         = _scannPqBits;
    copy.scannRerankK        = _scannRerankK;
    copy.scannAqEta          = _scannAqEta;
    return copy;
}

/** Convert to the C PistaDBParams struct. */
- (PistaDBParams)toCParams {
    PistaDBParams p = pistadb_default_params();
    p.hnsw_M               = (int)_hnswM;
    p.hnsw_ef_construction = (int)_hnswEfConstruction;
    p.hnsw_ef_search       = (int)_hnswEfSearch;
    p.ivf_nlist            = (int)_ivfNlist;
    p.ivf_nprobe           = (int)_ivfNprobe;
    p.pq_M                 = (int)_pqM;
    p.pq_nbits             = (int)_pqNbits;
    p.diskann_R            = (int)_diskannR;
    p.diskann_L            = (int)_diskannL;
    p.diskann_alpha        = _diskannAlpha;
    p.lsh_L                = (int)_lshL;
    p.lsh_K                = (int)_lshK;
    p.lsh_w                = _lshW;
    p.scann_nlist          = (int)_scannNlist;
    p.scann_nprobe         = (int)_scannNprobe;
    p.scann_pq_M           = (int)_scannPqM;
    p.scann_pq_bits        = (int)_scannPqBits;
    p.scann_rerank_k       = (int)_scannRerankK;
    p.scann_aq_eta         = _scannAqEta;
    return p;
}

@end

/* ── PSTDatabase ──────────────────────────────────────────────────────────── */

@implementation PSTDatabase {
    PistaDB *_db;       /* opaque native handle          */
    NSLock  *_lock;     /* serialises all native calls   */
}

/* ── Construction ─────────────────────────────────────────────────────────── */

+ (nullable instancetype)databaseWithPath:(NSString *)path
                                      dim:(NSInteger)dim
                                   metric:(PSTMetric)metric
                                indexType:(PSTIndexType)indexType
                                   params:(nullable PSTParams *)params
                                    error:(NSError *__autoreleasing _Nullable *)error
{
    return [[self alloc] initWithPath:path dim:dim metric:metric
                            indexType:indexType params:params error:error];
}

- (nullable instancetype)initWithPath:(NSString *)path
                                  dim:(NSInteger)dim
                               metric:(PSTMetric)metric
                            indexType:(PSTIndexType)indexType
                               params:(nullable PSTParams *)params
                                error:(NSError *__autoreleasing _Nullable *)error
{
    if (!(self = [super init])) return nil;

    PistaDBParams cp;
    const PistaDBParams *cpp = NULL;
    if (params) { cp = [params toCParams]; cpp = &cp; }

    _db = pistadb_open(path.UTF8String,
                       (int)dim,
                       (PistaDBMetric)metric,
                       (PistaDBIndexType)indexType,
                       cpp);
    if (!_db) {
        if (error) {
            *error = [NSError errorWithDomain:PSTDatabaseErrorDomain
                                         code:PSTDatabaseErrorUnknown
                                     userInfo:@{NSLocalizedDescriptionKey:
                                                    @"Failed to open database"}];
        }
        return nil;
    }

    _lock = [[NSLock alloc] init];
    _lock.name = @"com.pistadb.lock";
    return self;
}

- (void)dealloc {
    [self _closeNoSave];
}

/* ── Lifecycle ────────────────────────────────────────────────────────────── */

- (BOOL)save:(NSError *__autoreleasing _Nullable *)error {
    [_lock lock];
    BOOL ok = [self _saveUnlocked:error];
    [_lock unlock];
    return ok;
}

- (void)close {
    [_lock lock];
    [self _saveUnlocked:nil];
    [self _closeNoSave];
    [_lock unlock];
}

/* ── CRUD ─────────────────────────────────────────────────────────────────── */

- (BOOL)insertId:(uint64_t)vectorId
      floatArray:(const float *)floats
           count:(NSInteger)count
           label:(nullable NSString *)label
           error:(NSError *__autoreleasing _Nullable *)error
{
    [_lock lock];
    int rc = pistadb_insert(_db, vectorId, label.UTF8String, floats);
    [_lock unlock];
    return [self _check:rc error:error];
}

- (BOOL)deleteId:(uint64_t)vectorId
           error:(NSError *__autoreleasing _Nullable *)error
{
    [_lock lock];
    int rc = pistadb_delete(_db, vectorId);
    [_lock unlock];
    return [self _check:rc error:error];
}

- (BOOL)updateId:(uint64_t)vectorId
      floatArray:(const float *)floats
           count:(NSInteger)count
           error:(NSError *__autoreleasing _Nullable *)error
{
    [_lock lock];
    int rc = pistadb_update(_db, vectorId, floats);
    [_lock unlock];
    return [self _check:rc error:error];
}

- (nullable PSTVectorEntry *)entryForId:(uint64_t)vectorId
                                  error:(NSError *__autoreleasing _Nullable *)error
{
    [_lock lock];
    int dim = pistadb_dim(_db);
    [_lock unlock];

    float *buf = (float *)malloc((size_t)dim * sizeof(float));
    if (!buf) {
        if (error) *error = [self _oomError];
        return nil;
    }

    char label[256] = {0};
    [_lock lock];
    int rc = pistadb_get(_db, vectorId, buf, label);
    [_lock unlock];

    if (rc != PISTADB_OK) {
        free(buf);
        [self _check:rc error:error];
        return nil;
    }

    PSTVectorEntry *entry = [[PSTVectorEntry alloc]
                             initWithFloats:buf count:dim
                             label:@(label)];
    free(buf);
    return entry;
}

/* ── Search ───────────────────────────────────────────────────────────────── */

- (nullable NSArray<PSTSearchResult *> *)searchFloatArray:(const float *)query
                                                    count:(NSInteger)count
                                                        k:(NSInteger)k
                                                    error:(NSError *__autoreleasing _Nullable *)error
{
    PistaDBResult *results = (PistaDBResult *)malloc((size_t)k * sizeof(PistaDBResult));
    if (!results) {
        if (error) *error = [self _oomError];
        return nil;
    }

    [_lock lock];
    int n = pistadb_search(_db, query, (int)k, results);
    [_lock unlock];

    if (n < 0) {
        free(results);
        [self _check:n error:error];
        return nil;
    }

    NSMutableArray<PSTSearchResult *> *arr = [NSMutableArray arrayWithCapacity:(NSUInteger)n];
    for (int i = 0; i < n; i++) {
        [arr addObject:[[PSTSearchResult alloc]
                        initWithId:results[i].id
                          distance:results[i].distance
                             label:@(results[i].label)]];
    }
    free(results);
    return [arr copy];
}

/* ── Index management ─────────────────────────────────────────────────────── */

- (BOOL)train:(NSError *__autoreleasing _Nullable *)error {
    [_lock lock];
    int rc = pistadb_train(_db);
    [_lock unlock];
    return [self _check:rc error:error];
}

/* ── Metadata ─────────────────────────────────────────────────────────────── */

- (NSInteger)count {
    [_lock lock];
    int n = _db ? pistadb_count(_db) : 0;
    [_lock unlock];
    return n;
}

- (NSInteger)dim {
    [_lock lock];
    int d = _db ? pistadb_dim(_db) : 0;
    [_lock unlock];
    return d;
}

- (PSTMetric)metric {
    [_lock lock];
    PSTMetric m = _db ? (PSTMetric)pistadb_metric(_db) : PSTMetricL2;
    [_lock unlock];
    return m;
}

- (PSTIndexType)indexType {
    [_lock lock];
    PSTIndexType t = _db ? (PSTIndexType)pistadb_index_type(_db) : PSTIndexTypeLinear;
    [_lock unlock];
    return t;
}

- (NSString *)lastError {
    [_lock lock];
    NSString *msg = _db ? @(pistadb_last_error(_db)) : @"database is closed";
    [_lock unlock];
    return msg;
}

+ (NSString *)version {
    return @(pistadb_version());
}

/* ── Private ──────────────────────────────────────────────────────────────── */

/** Must be called with _lock held. */
- (BOOL)_saveUnlocked:(NSError *__autoreleasing _Nullable *)error {
    if (!_db) return YES;
    int rc = pistadb_save(_db);
    return [self _check:rc error:error];
}

- (void)_closeNoSave {
    if (_db) {
        pistadb_close(_db);
        _db = NULL;
    }
}

- (BOOL)_check:(int)rc error:(NSError *__autoreleasing _Nullable *)outError {
    if (rc == PISTADB_OK) return YES;
    if (outError) {
        NSString *msg = _db ? @(pistadb_last_error(_db)) : @"Unknown error";
        *outError = [NSError errorWithDomain:PSTDatabaseErrorDomain
                                        code:rc
                                    userInfo:@{NSLocalizedDescriptionKey: msg}];
    }
    return NO;
}

- (NSError *)_oomError {
    return [NSError errorWithDomain:PSTDatabaseErrorDomain
                               code:PSTDatabaseErrorNoMemory
                           userInfo:@{NSLocalizedDescriptionKey: @"Out of memory"}];
}

@end
