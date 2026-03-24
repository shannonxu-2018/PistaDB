/*
 *    ___ _    _        ___  ___
 *   | _ (_)__| |_ __ _|   \| _ )
 *   |  _/ (_-<  _/ _` | |) | _ \
 *   |_| |_/__/\__\__,_|___/|___/
 */
import Foundation

// All native calls are blocking (I/O + in-memory graph traversal).
// These async wrappers dispatch work to a background executor so they
// are safe to call from MainActor / SwiftUI without freezing the UI.

@available(iOS 13.0, macOS 10.15, watchOS 6.0, tvOS 13.0, *)
public extension PistaDB {

    // MARK: - Async insert

    /// Async variant of ``insert(id:vector:label:)``.
    func insert(id: UInt64, vector: [Float], label: String = "") async throws {
        try await _background { try self.insert(id: id, vector: vector, label: label) }
    }

    // MARK: - Async delete

    /// Async variant of ``delete(id:)``.
    func delete(id: UInt64) async throws {
        try await _background { try self.delete(id: id) }
    }

    // MARK: - Async update

    /// Async variant of ``update(id:vector:)``.
    func update(id: UInt64, vector: [Float]) async throws {
        try await _background { try self.update(id: id, vector: vector) }
    }

    // MARK: - Async get

    /// Async variant of ``get(id:)``.
    func get(id: UInt64) async throws -> VectorEntry {
        try await _background { try self.get(id: id) }
    }

    // MARK: - Async search

    /// Async variant of ``search(_:k:)``.
    ///
    /// ```swift
    /// let results = try await db.search(queryEmbedding, k: 10)
    /// ```
    func search(_ query: [Float], k: Int = 10) async throws -> [SearchResult] {
        try await _background { try self.search(query, k: k) }
    }

    // MARK: - Async batch insert

    /// Async variant of ``insertBatch(_:)``.
    func insertBatch<S: Sequence>(_ entries: S) async throws
        where S.Element == (id: UInt64, vector: [Float], label: String)
    {
        // Materialise into an Array first so we can safely cross the concurrency
        // boundary without iterator state living on the caller's actor.
        let batch = Array(entries)
        try await _background { try self.insertBatch(batch) }
    }

    // MARK: - Async train

    /// Async variant of ``train()``.
    func train() async throws {
        try await _background { try self.train() }
    }

    // MARK: - Async save

    /// Async variant of ``save()``.
    func save() async throws {
        try await _background { try self.save() }
    }
}

// MARK: - Internal

@available(iOS 13.0, macOS 10.15, watchOS 6.0, tvOS 13.0, *)
private func _background<T: Sendable>(_ work: @escaping () throws -> T) async throws -> T {
    try await withCheckedThrowingContinuation { cont in
        DispatchQueue.global(qos: .userInitiated).async {
            do    { cont.resume(returning: try work()) }
            catch { cont.resume(throwing: error) }
        }
    }
}
