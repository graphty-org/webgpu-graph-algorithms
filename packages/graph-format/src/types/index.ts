/**
 * Barrel of the public type surface (design section 12.2), split by concern across columns.ts
 * (scalars, typed-array aliases, columns, tables, metadata, masks), snapshot.ts (id map, flags,
 * arena, views, derived graphs, the snapshot contract, factory inputs), builder.ts (builder options,
 * freeze report, handles, GraphSink, the builder contract) and wire.ts (the wire form). internal.ts
 * is deliberately absent: its construction contracts are shared by the implementation modules only.
 *
 * Later modules import types from here: `import { type U32, type Column } from "../types/index.js"`.
 */

export type * from "./builder.js";
export type * from "./columns.js";
export type * from "./snapshot.js";
export type * from "./wire.js";
