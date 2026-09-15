/**
 * Code-quality audit of the source tree by mechanical scan: functions longer than 150 lines,
 * helpers duplicated across the format directories that belong in src/common, and the two
 * record / XML readers that coexist. The scans read src/ through the TypeScript compiler API
 * and plain text; they do not import the modules.
 *
 * Every test here fails deliberately and pins a finding of the audit report:
 * - seven functions exceed 150 lines (json/exporter.ts `plan` 237, graphml/exporter.ts
 *   `planExport` 232, pajek/exporter.ts `writeParts` 184, csv/exporter.ts `planExport` 183,
 *   gexf/xml.ts `scanXmlSyntax` 176, gexf/exporter.ts `collectAttributes` 170, json/importer.ts
 *   `importCytoscape` 164);
 * - "is edge e the mirror half of an expanded pair" is implemented in eight exporters, "the
 *   explicit weight text of edge e" in seven, "an f64 text with a decimal point guaranteed" in
 *   four, the XML entity decoder and NAMED_ENTITIES table in two, `isSpace` in three, and
 *   `E_CSV_UNCLOSED_QUOTE` is defined by two independent CSV record readers (csv/records.ts on
 *   papaparse, neo4j/records.ts hand-written).
 */

import { readdirSync, readFileSync, statSync } from "node:fs";
import { dirname, join, relative } from "node:path";
import { fileURLToPath } from "node:url";

import ts from "typescript";
import { describe, expect, it } from "vitest";

const SRC = join(dirname(fileURLToPath(import.meta.url)), "..", "..", "src");
const LIMIT = 150;

function walk(dir: string, out: string[] = []): string[] {
    for (const name of readdirSync(dir)) {
        const path = join(dir, name);
        if (statSync(path).isDirectory()) {
            walk(path, out);
        } else if (path.endsWith(".ts")) {
            out.push(path);
        }
    }
    return out;
}

interface LongFunction {
    readonly file: string;
    readonly name: string;
    readonly lines: number;
}

function longFunctions(file: string): LongFunction[] {
    const text = readFileSync(file, "utf8");
    const sf = ts.createSourceFile(file, text, ts.ScriptTarget.Latest, true);
    const found: LongFunction[] = [];
    const visit = (node: ts.Node): void => {
        if (
            (ts.isFunctionDeclaration(node) ||
                ts.isMethodDeclaration(node) ||
                ts.isFunctionExpression(node) ||
                ts.isArrowFunction(node) ||
                ts.isConstructorDeclaration(node) ||
                ts.isGetAccessor(node)) &&
            node.body !== undefined
        ) {
            const start = sf.getLineAndCharacterOfPosition(node.getStart(sf)).line;
            const end = sf.getLineAndCharacterOfPosition(node.end).line;
            const lines = end - start + 1;
            if (lines > LIMIT) {
                const name = ts.isConstructorDeclaration(node)
                    ? "constructor"
                    : (node.name?.getText(sf) ?? "<anonymous>");
                found.push({ file: relative(SRC, file), name, lines });
            }
        }
        ts.forEachChild(node, visit);
    };
    visit(sf);
    return found;
}

/** The files under src/formats whose text matches a pattern, relative to src/. */
function formatFilesMatching(pattern: RegExp): string[] {
    return walk(join(SRC, "formats"))
        .filter((file) => pattern.test(readFileSync(file, "utf8")))
        .map((file) => relative(SRC, file))
        .sort();
}

describe("audit: source metrics", () => {
    it(`no function is longer than ${LIMIT} lines`, () => {
        const found = walk(SRC)
            .flatMap(longFunctions)
            .sort((a, b) => b.lines - a.lines);
        expect(found.map((f) => `${f.file} ${f.name} (${f.lines} lines)`)).toEqual([]);
    });

    it("pair folding (mirror detection) is implemented once", () => {
        // every exporter that folds expanded pairs re-implements "pair[e] < e" itself
        const files = formatFilesMatching(
            /\b(isMirror|planMirror)\b|pairs?(Column)?\.data\[e\] < e|\(pairColumn\.value\(e\) as number\) < e|\bother < e\b|\bmate < e\b|return p !== INVALID_INDEX && p < e/,
        );
        expect(files).toEqual([]);
    });

    it("the explicit-weight-text rule is implemented once", () => {
        const files = formatFilesMatching(
            /function weightText\(|private weightText\(|function weightAt\(|function explicitWeight\(|private planWeights\(/,
        );
        expect(files).toEqual([]);
    });

    it("the decimal-point-guaranteed float text is implemented once (formatGmlReal exists in common)", () => {
        const files = formatFilesMatching(/`\$\{text\}\.0`/);
        expect(files).toEqual([]);
    });

    it("XML entity decoding and whitespace tests exist once", () => {
        expect(formatFilesMatching(/const NAMED_ENTITIES/)).toEqual([]);
        expect(formatFilesMatching(/function isSpace\(/)).toEqual([]);
    });

    it("one CSV record reader defines the quote codes", () => {
        expect(formatFilesMatching(/"E_CSV_UNCLOSED_QUOTE"/)).toEqual(["formats/csv/records.ts"]);
    });
});
