/**
 * The names the DOT importer and exporter share: the format name, the column origin, the
 * format-owned columns (design section 5.6: `graphty.` prefix) and the attribute names the format
 * gives a fixed meaning to.
 */

import { type ColumnOriginInput } from "@graphty/graph-format";

/** The format name. */
export const DOT_FORMAT = "dot";

/** The origin recorded on every column the importer declares. */
export const DOT_ORIGIN: ColumnOriginInput = Object.freeze({ format: DOT_FORMAT });

/** The bool node column marking a cluster's container node. */
export const CLUSTER_COLUMN = "graphty.cluster";

/** The u32 node column (role parent, refersTo node) holding a member's cluster. */
export const PARENT_COLUMN = "graphty.parent";

/** The string edge column (role sourcePort) holding an edge's tail port. */
export const SOURCE_PORT_COLUMN = "graphty.sourcePort";

/** The string edge column (role targetPort) holding an edge's head port. */
export const TARGET_PORT_COLUMN = "graphty.targetPort";

/** The attribute that is the display label (kept as text with role label). */
export const LABEL_ATTRIBUTE = "label";

/** The node attribute holding a position, mapped to the position role column (design section 5.2). */
export const POS_ATTRIBUTE = "pos";

/** The edge attribute cgraph uses as an edge's identity within its endpoints, mapped to the edge id role. */
export const KEY_ATTRIBUTE = "key";

/** The node attribute a `!` suffix of `pos` sets (Graphviz's own spelling of a pinned position). */
export const PIN_ATTRIBUTE = "pin";

/** The key under meta.extra holding the format's graph flags (`{ strict: true }`). */
export const DOT_META_KEY = "dot";
