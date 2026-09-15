import * as graphIo from "@graphty/graph-io";
import { expectTypeOf } from "vitest";

// The package is ESM with named exports only: no default export.
expectTypeOf(graphIo).toBeObject();
expectTypeOf(graphIo).not.toHaveProperty("default");
