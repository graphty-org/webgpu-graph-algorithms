import { GraphBuilder, type GraphSnapshot, INVALID_INDEX } from "@graphty/graph-format";
import { describe, expect, it } from "vitest";

import {
    DIRECTED_COLUMN,
    DIRECTION_FORCED_CODE,
    DIRECTION_REFUSED_CODE,
    DirectionResolver,
    MIXED_DIRECTION_CODE,
    MUTUAL_COLUMN,
    PAIR_COLUMN,
} from "../../src/common/direction.js";
import { ImportReportBuilder } from "../../src/common/report.js";
import { ImportError } from "../../src/types.js";

type Policy = "expand" | "directed" | "undirected" | "error";

function setup(
    directed: boolean,
    policy: Policy = "expand",
    errorLimit = 100,
): {
    sink: GraphBuilder;
    report: ImportReportBuilder;
    resolver: DirectionResolver;
} {
    const sink = new GraphBuilder({ directed, weightDtype: "f64" });
    const report = new ImportReportBuilder("test", errorLimit);
    const resolver = new DirectionResolver(sink, report, policy);
    return { sink, report, resolver };
}

function directedFlags(s: GraphSnapshot): (boolean | undefined)[] {
    const column = s.edges.byRole("directed");
    return column === null
        ? []
        : Array.from({ length: s.edgeCount }, (_, e) => (column.isSet(e) ? (column.value(e) as boolean) : undefined));
}

function pairs(s: GraphSnapshot): (number | undefined)[] {
    const column = s.edges.byRole("pair");
    return column === null
        ? []
        : Array.from({ length: s.edgeCount }, (_, e) => (column.isSet(e) ? (column.value(e) as number) : undefined));
}

describe("DirectionResolver rule 1: the header", () => {
    it("sets an empty unlocked sink to the file's direction", () => {
        const { sink, resolver, report } = setup(true);
        resolver.setHeader(false);
        expect(sink.directed).toBe(false);
        expect(resolver.directed).toBe(false);
        expect(report.issues).toEqual([]);
        const other = setup(false);
        other.resolver.setHeader(true);
        expect(other.sink.directed).toBe(true);
    });

    it("records a coercion and keeps the sink's direction when the sink is locked", () => {
        const { sink, resolver, report } = setup(true);
        sink.lockDirected();
        resolver.setHeader(false);
        expect(sink.directed).toBe(true);
        expect(report.issues).toHaveLength(1);
        expect(report.issues[0]).toMatchObject({
            category: "coercion",
            severity: "warning",
            code: DIRECTION_REFUSED_CODE,
        });
        expect(report.issues[0].message).toContain("locked");
    });

    it("records a coercion for an undirected header on a non-empty directed sink", () => {
        const { sink, resolver, report } = setup(true);
        sink.addEdge("x", "y");
        resolver.setHeader(false);
        expect(sink.directed).toBe(true);
        expect(report.issues.map((i) => i.code)).toEqual([DIRECTION_REFUSED_CODE]);
    });

    it("expands a non-empty unlocked undirected sink for a directed header (a second file under auto)", () => {
        const { sink, resolver, report } = setup(false);
        sink.addEdge("x", "y", 2);
        sink.addEdge("y", "z");
        resolver.setHeader(true);
        expect(sink.directed).toBe(true);
        expect(sink.edgeCount).toBe(4);
        expect(report.counts.expandedMixed).toBe(2);
        expect(resolver.expanded).toBe(true);
        expect(report.issues).toEqual([]);
        // the new file's directed edges are marked directed
        resolver.addEdge("a", "b", "directed");
        const s = sink.freeze();
        expect(directedFlags(s)).toEqual([false, false, false, false, true]);
        expect(pairs(s)).toEqual([2, 3, 0, 1, undefined]);
    });

    it("forces the policy's direction under directed / undirected and reports the difference", () => {
        const d = setup(false, "directed");
        d.resolver.setHeader(false);
        expect(d.sink.directed).toBe(true);
        expect(d.report.issues.map((i) => i.code)).toEqual([DIRECTION_FORCED_CODE]);
        const u = setup(true, "undirected");
        u.resolver.setHeader(true);
        expect(u.sink.directed).toBe(false);
        expect(u.report.issues.map((i) => i.code)).toEqual([DIRECTION_FORCED_CODE]);
        const agree = setup(true, "directed");
        agree.resolver.setHeader(true);
        expect(agree.report.issues).toEqual([]);
    });

    it("adopts reserved columns a caller's sink already holds", () => {
        const { sink, resolver } = setup(false);
        sink.addEdge("x", "y");
        sink.setDirected(true, { expand: true });
        resolver.setHeader(true);
        expect(resolver.expanded).toBe(true);
        resolver.addEdge("a", "b", "directed");
        expect(directedFlags(sink.freeze())).toEqual([false, false, true]);
    });

    it("refuses addEdge before setHeader", () => {
        const { resolver } = setup(true);
        expect(() => resolver.addEdge("a", "b", "directed")).toThrow(/setHeader/);
    });
});

describe("DirectionResolver rule 2: expand", () => {
    it("pushes edges that agree with the sink once, without reserved columns", () => {
        const { sink, resolver, report } = setup(true);
        resolver.setHeader(true);
        expect(resolver.addEdge("a", "b", "directed", 2)).toBe(0);
        expect(resolver.addEdge("b", "c", "directed")).toBe(1);
        const s = sink.freeze();
        expect(s.edgeCount).toBe(2);
        expect(s.edges.byRole("pair")).toBeNull();
        expect(s.edges.byRole("directed")).toBeNull();
        expect(resolver.expanded).toBe(false);
        expect(report.counts.expandedMixed).toBe(0);
        const u = setup(false);
        u.resolver.setHeader(false);
        u.resolver.addEdge("a", "b", "undirected");
        expect(u.sink.freeze().edges.byRole("pair")).toBeNull();
    });

    it("expands an undirected edge into a directed sink as a linked pair and backfills earlier edges", () => {
        const { sink, resolver, report } = setup(true);
        resolver.setHeader(true);
        resolver.addEdge("a", "b", "directed", 5);
        const primary = resolver.addEdge("b", "c", "undirected", 7);
        expect(primary).toBe(1);
        resolver.addEdge("c", "a", "directed");
        const s = sink.freeze();
        expect(s.directed).toBe(true);
        expect(s.edgeCount).toBe(4);
        expect(directedFlags(s)).toEqual([true, false, false, true]);
        expect(pairs(s)).toEqual([undefined, 2, 1, undefined]);
        const list = s.edgeList();
        expect([list.src[1], list.dst[1]]).toEqual([1, 2]);
        expect([list.src[2], list.dst[2]]).toEqual([2, 1]);
        expect(list.weights?.[1]).toBe(7);
        expect(list.weights?.[2]).toBe(7);
        expect(report.counts.expandedMixed).toBe(1);
        expect(report.issues).toEqual([]);
        const pair = s.edges.byRole("pair");
        expect(pair?.meta.refersTo).toBe("edge");
        expect(pair?.meta.name).toBe(PAIR_COLUMN);
        expect(s.edges.byRole("directed")?.meta.name).toBe(DIRECTED_COLUMN);
    });

    it("does not mirror an undirected self-loop", () => {
        const { sink, resolver, report } = setup(true);
        resolver.setHeader(true);
        resolver.addEdge("a", "a", "undirected");
        const s = sink.freeze();
        expect(s.edgeCount).toBe(1);
        expect(directedFlags(s)).toEqual([false]);
        expect(pairs(s)).toEqual([undefined]);
        expect(report.counts.expandedMixed).toBe(1);
    });

    it("keeps an absent weight absent on both halves", () => {
        const { sink, resolver } = setup(true);
        resolver.setHeader(true);
        resolver.addEdge("a", "b", "directed", 3);
        resolver.addEdge("b", "c", "undirected");
        const s = sink.freeze();
        const shadow = s.edges.byRole("weight");
        expect(shadow?.isSet(0)).toBe(true);
        expect(shadow?.isSet(1)).toBe(false);
        expect(shadow?.isSet(2)).toBe(false);
    });

    it("expands the sink once when a directed edge reaches an undirected sink and continues directed", () => {
        const { sink, resolver, report } = setup(false);
        resolver.setHeader(false);
        resolver.addEdge("a", "b", "undirected", 1.5);
        resolver.addEdge("b", "c", "undirected");
        resolver.addEdge("c", "c", "undirected");
        expect(sink.directed).toBe(false);
        const e = resolver.addEdge("c", "d", "directed", 9);
        expect(sink.directed).toBe(true);
        expect(resolver.expanded).toBe(true);
        expect(report.counts.expandedMixed).toBe(3);
        resolver.addEdge("d", "e", "undirected");
        expect(report.counts.expandedMixed).toBe(4);
        const s = sink.freeze();
        expect(s.edgeCount).toBe(3 + 2 + 1 + 2);
        expect(directedFlags(s)[e]).toBe(true);
        const list = s.edgeList();
        expect([list.src[e], list.dst[e]]).toEqual([2, 3]);
        expect(list.weights?.[e]).toBe(9);
        expect(directedFlags(s).filter((f) => f === false)).toHaveLength(7);
        expect(report.issues).toEqual([]);
    });

    it("aborts with E_IMPORT when a locked undirected sink cannot be expanded", () => {
        const { sink, resolver, report } = setup(false);
        sink.lockDirected();
        resolver.setHeader(false);
        resolver.addEdge("a", "b", "undirected");
        let caught: unknown;
        try {
            resolver.addEdge("b", "c", "directed");
        } catch (err) {
            caught = err;
        }
        expect(caught).toBeInstanceOf(ImportError);
        const err = caught as ImportError;
        expect(err.report.issues.map((i) => [i.category, i.severity, i.code])).toEqual([
            ["coercion", "error", DIRECTION_REFUSED_CODE],
        ]);
        expect(err.details.code).toBe(DIRECTION_REFUSED_CODE);
        expect(report.errorCount).toBe(1);
        expect(sink.edgeCount).toBe(1);
    });

    it("expands mutual edges as a pair marked directed with the mutual role on the primary", () => {
        const { sink, resolver } = setup(true);
        resolver.setHeader(true);
        resolver.addEdge("a", "b", "directed");
        const m = resolver.addEdge("b", "c", "mutual", 2);
        const s = sink.freeze();
        expect(s.edgeCount).toBe(3);
        expect(directedFlags(s)).toEqual([true, true, true]);
        expect(pairs(s)).toEqual([undefined, 2, 1]);
        const mutual = s.edges.byRole("mutual");
        expect(mutual?.meta.name).toBe(MUTUAL_COLUMN);
        expect(mutual?.isSet(m)).toBe(true);
        expect(mutual?.value(m)).toBe(true);
        expect(mutual?.isSet(2)).toBe(false);
        expect(mutual?.isSet(0)).toBe(false);
        expect(s.edgeList().weights?.[2]).toBe(2);
    });

    it("a mutual edge into an undirected sink expands the sink first", () => {
        const { sink, resolver } = setup(false);
        resolver.setHeader(false);
        resolver.addEdge("a", "b", "undirected");
        resolver.addEdge("b", "c", "mutual");
        expect(sink.directed).toBe(true);
        const s = sink.freeze();
        expect(s.edgeCount).toBe(4);
        expect(directedFlags(s)).toEqual([false, false, true, true]);
        expect(pairs(s)).toEqual([1, 0, 3, 2]);
    });
});

describe("DirectionResolver policies directed / undirected / error", () => {
    it("directed: every edge is one directed edge, reported once with the count", () => {
        const { sink, resolver, report } = setup(true, "directed");
        resolver.setHeader(false);
        resolver.addEdge("a", "b", "undirected");
        resolver.addEdge("b", "c", "undirected");
        resolver.addEdge("c", "d", "directed");
        resolver.addEdge("d", "a", "mutual");
        const s = sink.freeze();
        expect(s.directed).toBe(true);
        expect(s.edgeCount).toBe(4);
        expect(s.edges.byRole("pair")).toBeNull();
        expect(resolver.forced).toBe(3);
        expect(report.counts.expandedMixed).toBe(0);
        const codes = report.issues.map((i) => i.code);
        expect(codes.filter((c) => c === DIRECTION_FORCED_CODE)).toHaveLength(2);
    });

    it("undirected: every edge is one undirected edge", () => {
        const { sink, resolver, report } = setup(false, "undirected");
        resolver.setHeader(true);
        resolver.addEdge("a", "b", "directed");
        resolver.addEdge("b", "c", "undirected");
        const s = sink.freeze();
        expect(s.directed).toBe(false);
        expect(s.edgeCount).toBe(2);
        expect(resolver.forced).toBe(1);
        expect(report.warningCount).toBe(2);
    });

    it("undirected against a locked directed sink reports and pushes as the sink's kind", () => {
        const { sink, resolver, report } = setup(true, "undirected");
        sink.lockDirected();
        resolver.setHeader(true);
        resolver.addEdge("a", "b", "directed");
        resolver.addEdge("b", "c", "undirected");
        expect(sink.directed).toBe(true);
        expect(sink.edgeCount).toBe(2);
        expect(report.issues.map((i) => i.code)).toEqual([
            DIRECTION_FORCED_CODE,
            DIRECTION_REFUSED_CODE,
            DIRECTION_FORCED_CODE,
        ]);
        expect(resolver.forced).toBe(1);
    });

    it("error: the first differing edge aborts with E_IMPORT and a validation-error", () => {
        const { sink, resolver } = setup(true, "error");
        resolver.setHeader(true);
        resolver.addEdge("a", "b", "directed");
        let caught: unknown;
        try {
            resolver.addEdge("b", "c", "undirected", undefined, { line: 12, element: "e1" });
        } catch (err) {
            caught = err;
        }
        expect(caught).toBeInstanceOf(ImportError);
        const err = caught as ImportError;
        expect(err.report.issues).toEqual([
            {
                category: "validation-error",
                severity: "error",
                code: MIXED_DIRECTION_CODE,
                message: expect.stringContaining("undirected edge in a directed graph") as string,
                line: 12,
                element: "e1",
            },
        ]);
        expect(sink.edgeCount).toBe(1);
    });

    it("error: agreeing edges are fine", () => {
        const { sink, resolver } = setup(false, "error");
        resolver.setHeader(false);
        resolver.addEdge("a", "b", "undirected");
        expect(sink.edgeCount).toBe(1);
    });
});

describe("DirectionResolver and the report counts", () => {
    it("returns builder edge indices and never touches nodes / edges counters itself", () => {
        const { sink, resolver, report } = setup(true);
        resolver.setHeader(true);
        const e0 = resolver.addEdge("a", "b", "directed");
        const e1 = resolver.addEdge("b", "c", "undirected");
        expect(e0).toBe(0);
        expect(e1).toBe(1);
        expect(sink.indexOf("a")).not.toBe(INVALID_INDEX);
        expect(report.counts.edges).toBe(0);
        expect(report.counts.nodes).toBe(0);
        expect(report.counts.expandedMixed).toBe(1);
    });
});
