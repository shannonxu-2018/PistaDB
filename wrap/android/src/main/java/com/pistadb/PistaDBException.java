/*
 *    ___ _    _        ___  ___
 *   | _ (_)__| |_ __ _|   \| _ )
 *   |  _/ (_-<  _/ _` | |) | _ \
 *   |_| |_/__/\__\__,_|___/|___/
 */
package com.pistadb;

/**
 * Thrown when a PistaDB operation fails.
 * The message contains the underlying C-level error description.
 */
public class PistaDBException extends RuntimeException {

    public PistaDBException(String message) {
        super(message);
    }

    public PistaDBException(String message, Throwable cause) {
        super(message, cause);
    }
}
