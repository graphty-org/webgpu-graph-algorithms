import * as graphFormat from "@graphty/graph-format";
import { expectTypeOf } from "vitest";

// The package is ESM with named exports only (design section 12.2): no default export.
expectTypeOf(graphFormat).toBeObject();
expectTypeOf(graphFormat).not.toHaveProperty("default");
