#    ___ _    _        ___  ___
#   | _ (_)__| |_ __ _|   \| _ )
#   |  _/ (_-<  _/ _` | |) | _ \
#   |_| |_/__/\__\__,_|___/|___/

# PistaDB – consumer ProGuard / R8 rules
# Applied automatically when this library is consumed via Gradle.

# Keep all public API classes
-keep public class com.pistadb.PistaDB        { public *; }
-keep public class com.pistadb.SearchResult   { public *; }
-keep public class com.pistadb.VectorEntry    { public *; }
-keep public class com.pistadb.PistaDBParams  { public *; *; }
-keep public class com.pistadb.PistaDBException { *; }
-keep public enum  com.pistadb.Metric         { *; }
-keep public enum  com.pistadb.IndexType      { *; }

# Keep PistaDBParams fields – JNI reads them by name via reflection
-keepclassmembers class com.pistadb.PistaDBParams {
    public int    hnswM;
    public int    hnswEfConstruction;
    public int    hnswEfSearch;
    public int    ivfNlist;
    public int    ivfNprobe;
    public int    pqM;
    public int    pqNbits;
    public int    diskannR;
    public int    diskannL;
    public float  diskannAlpha;
    public int    lshL;
    public int    lshK;
    public float  lshW;
    public int    scannNlist;
    public int    scannNprobe;
    public int    scannPqM;
    public int    scannPqBits;
    public int    scannRerankK;
    public float  scannAqEta;
}

# Keep JNI-called constructors (JNI creates these with NewObject)
-keepclassmembers class com.pistadb.SearchResult {
    public <init>(long, float, java.lang.String);
}
-keepclassmembers class com.pistadb.VectorEntry {
    public <init>(float[], java.lang.String);
}
