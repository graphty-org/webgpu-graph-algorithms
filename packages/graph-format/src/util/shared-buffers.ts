/**
 * The conservative owner count of design section 9.1. The format shares storage deliberately:
 * `withColumns()` shares the core, same-node-space derived graphs share the node table and the id
 * map, `clone()` and `set(name, column)` share Column objects between tables, `transpose()` adopts
 * the cached reverse arrays as its core. Every holder of a buffer, column, table or id map claims it
 * once (the snapshot and table constructors do so); a target claimed by more than one holder is
 * SHARED, and `toWire({ transfer: true })` / `transferables()` copy its buffers instead of
 * transferring them, so no sibling snapshot is ever silently emptied. The count is never decremented
 * (conservative: a holder that is garbage-collected still counts, which only ever causes a copy).
 *
 * Holders are tracked at three granularities because a shared holder materialises buffers lazily
 * (the Utf8 store of a string column or a string id map, the Float64Array of a numeric id map): a
 * buffer, a Column (or a NodeIdMap) and an AttributeTable. The wire module treats a buffer as shared
 * when the buffer itself, the column that holds it or the table that holds the column is shared.
 * This module is a leaf (it imports nothing) so every module may use it without a cycle.
 */

/** Target (ArrayBuffer, Column, AttributeTable or NodeIdMap) -> number of holders recorded. */
const HOLDERS = new WeakMap<object, number>();

/**
 * Record one holder of a buffer, column, table or id map. Called by every constructor that keeps a
 * reference to storage it did not allocate itself; the second claim of the same target marks it
 * shared.
 * @param target - the buffer, Column, AttributeTable or NodeIdMap now held
 */
export function claimHolder(target: object): void {
    HOLDERS.set(target, (HOLDERS.get(target) ?? 0) + 1);
}

/**
 * Record that a target has gained another holder (design section 9.1), so it is treated as shared
 * from now on whatever its previous count.
 * @param target - the buffer, Column, AttributeTable or NodeIdMap that is now held by more than one holder
 */
export function noteShared(target: object): void {
    HOLDERS.set(target, Math.max(HOLDERS.get(target) ?? 0, 1) + 1);
}

/**
 * The number of holders recorded for a target.
 * @param target - the buffer, Column, AttributeTable or NodeIdMap
 * @returns the count, 0 when never claimed
 */
export function holderCount(target: object): number {
    return HOLDERS.get(target) ?? 0;
}

/**
 * Whether a target has been recorded as held by more than one holder.
 * @param target - the buffer, Column, AttributeTable or NodeIdMap
 * @returns true when at least two holders were recorded
 */
export function isShared(target: object): boolean {
    return (HOLDERS.get(target) ?? 0) > 1;
}
