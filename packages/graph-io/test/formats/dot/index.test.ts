import { describe, expect, it } from "vitest";

import * as dot from "../../../src/formats/dot/index.js";

describe("the dot subpath entry", () => {
    it("re-exports the importer, the exporter and their codes", () => {
        expect(Object.keys(dot).sort()).toEqual(["DOT_ISSUE", "DOT_LOSS", "dotExporter", "dotImporter"]);
        expect(dot.dotImporter.format).toBe("dot");
        expect(dot.dotExporter.format).toBe("dot");
        expect(dot.DOT_ISSUE.SYNTAX).toBe("E_SYNTAX");
        expect(dot.DOT_LOSS.TRAILING_BACKSLASH).toBe("E_DOT_TRAILING_BACKSLASH");
    });
});
