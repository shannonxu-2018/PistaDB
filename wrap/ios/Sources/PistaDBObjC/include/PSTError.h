/*
 *    ___ _    _        ___  ___
 *   | _ (_)__| |_ __ _|   \| _ )
 *   |  _/ (_-<  _/ _` | |) | _ \
 *   |_| |_/__/\__\__,_|___/|___/
 */
#import <Foundation/Foundation.h>

NS_ASSUME_NONNULL_BEGIN

/** NSError domain for all PSTDatabase errors. */
extern NSErrorDomain const PSTDatabaseErrorDomain;

/**
 * Error codes matching the C PISTADB_E* constants.
 * Use these to distinguish failure types in an NSError.
 *
 * In Swift these cases are accessed as PSTDatabaseErrorCode with prefix
 * "PSTDatabaseError" stripped, e.g. PSTDatabaseErrorCode.notFound.
 */
typedef NS_ENUM(NSInteger, PSTDatabaseErrorCode) {
    PSTDatabaseErrorUnknown       = -1,  /**< Generic / unclassified error.    */
    PSTDatabaseErrorNoMemory      = -2,  /**< Allocation failure.              */
    PSTDatabaseErrorIO            = -3,  /**< I/O error during read/write.     */
    PSTDatabaseErrorNotFound      = -4,  /**< Vector id not found.             */
    PSTDatabaseErrorAlreadyExists = -5,  /**< Vector id already inserted.      */
    PSTDatabaseErrorInvalidArg    = -6,  /**< Invalid argument (dim, k, …).   */
    PSTDatabaseErrorNotTrained    = -7,  /**< Index not trained yet.           */
    PSTDatabaseErrorCorrupt       = -8,  /**< Corrupted .pst file.             */
    PSTDatabaseErrorVersion       = -9,  /**< Incompatible file version.       */
};

NS_ASSUME_NONNULL_END
