/**
 * Corpus access for the Neo4j suites: test/corpus/neo4j (written for graph-io, since no public
 * corpus exists for neo4j-admin CSV) with a manifest of the same shape as the other formats plus
 * two optional per-entry fields (`options` for importer options the file needs, `with` for a
 * companion relationship file and the counts of the paired import), and the malformed cases under
 * test/corpus/malformed/neo4j.
 */

import { existsSync, readdirSync, readFileSync } from "node:fs";
import { join } from "node:path";

import { type Neo4jImportOptions } from "../../../src/formats/neo4j/importer.js";
import { type CommonImportOptions } from "../../../src/types.js";
import { CORPUS_ROOT, type CorpusFile, type CorpusManifest } from "../../helpers/corpus.js";

/** A manifest entry with the Neo4j-specific optional fields. */
export interface Neo4jCorpusFile extends CorpusFile {
    /** Importer options the file needs (a tab delimiter, for instance). */
    readonly options?: (Neo4jImportOptions & CommonImportOptions) | undefined;
    /** A paired import: relationship files read after this one, and the expected counts of the pair. */
    readonly with?:
        | {
              readonly relationships: readonly string[];
              readonly expectedNodes: number;
              readonly expectedEdges: number;
          }
        | undefined;
}

const NEO4J_DIR = join(CORPUS_ROOT, "neo4j");
const MALFORMED_DIR = join(CORPUS_ROOT, "malformed", "neo4j");

/**
 * The Neo4j manifest.
 * @returns the parsed manifest
 */
export function neo4jManifest(): CorpusManifest & { readonly files: readonly Neo4jCorpusFile[] } {
    return JSON.parse(readFileSync(join(NEO4J_DIR, "manifest.json"), "utf-8")) as CorpusManifest & {
        readonly files: readonly Neo4jCorpusFile[];
    };
}

/**
 * The manifest entries.
 * @returns the files in manifest order
 */
export function neo4jFiles(): readonly Neo4jCorpusFile[] {
    return neo4jManifest().files;
}

/**
 * A corpus file as text.
 * @param name - the file name
 * @returns the UTF-8 text
 */
export function neo4jText(name: string): string {
    return readFileSync(join(NEO4J_DIR, name), "utf-8");
}

/**
 * A corpus file as bytes.
 * @param name - the file name
 * @returns the raw bytes
 */
export function neo4jBytes(name: string): Uint8Array {
    return new Uint8Array(readFileSync(join(NEO4J_DIR, name)));
}

/**
 * The malformed cases.
 * @returns the file names, sorted
 */
export function neo4jMalformedFiles(): readonly string[] {
    return existsSync(MALFORMED_DIR) ? readdirSync(MALFORMED_DIR).sort() : [];
}

/**
 * A malformed case as bytes.
 * @param name - the file name
 * @returns the raw bytes
 */
export function neo4jMalformedBytes(name: string): Uint8Array {
    return new Uint8Array(readFileSync(join(MALFORMED_DIR, name)));
}
