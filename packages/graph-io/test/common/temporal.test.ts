import { GraphFormatError } from "@graphty/graph-format";
import { describe, expect, it } from "vitest";

import {
    formatTemporal,
    formatTimeValue,
    parseTemporal,
    parseTimeText,
    TIME_TEXT_ROLE,
    TIME_TEXT_SUFFIX,
    timeTextCompanion,
} from "../../src/common/temporal.js";

function codeOf(fn: () => unknown): string | null {
    try {
        fn();
    } catch (err) {
        return err instanceof GraphFormatError ? err.code : "other";
    }
    return null;
}

describe("parseTemporal / formatTemporal (design 5.1)", () => {
    it("parses dates to UTC midnight epoch ms with no companion when canonical", () => {
        expect(parseTemporal("2009-03-01", "date")).toEqual({ value: Date.UTC(2009, 2, 1), text: null });
        expect(formatTemporal(Date.UTC(2009, 2, 1), "date")).toBe("2009-03-01");
    });

    it("keeps the source text when the canonical form differs", () => {
        // a date with a time part in a date column
        expect(parseTemporal("2009-03-01T10:00:00Z", "date")).toEqual({
            value: Date.UTC(2009, 2, 1, 10),
            text: "2009-03-01T10:00:00Z",
        });
        // a UTC offset
        expect(parseTemporal("2009-03-01T12:00:00+02:00", "dateTime")).toEqual({
            value: Date.UTC(2009, 2, 1, 10),
            text: "2009-03-01T12:00:00+02:00",
        });
        // fractional seconds beyond milliseconds
        expect(parseTemporal("2009-03-01T12:00:00.123456Z", "dateTime")).toEqual({
            value: Date.UTC(2009, 2, 1, 12, 0, 0, 123),
            text: "2009-03-01T12:00:00.123456Z",
        });
        // a dateTime without a zone
        expect(parseTemporal("2009-03-01T12:00:00", "dateTime")).toEqual({
            value: Date.UTC(2009, 2, 1, 12),
            text: "2009-03-01T12:00:00",
        });
        // a date without a time in a dateTime column
        expect(parseTemporal("2009-03-01", "dateTime")).toEqual({ value: Date.UTC(2009, 2, 1), text: "2009-03-01" });
    });

    it("round-trips canonical dateTime, localDateTime, time and localTime forms exactly", () => {
        for (const [text, kind] of [
            ["2009-03-01T12:34:56Z", "dateTime"],
            ["2009-03-01T12:34:56.500Z", "dateTime"],
            ["2009-03-01T12:34:56", "localDateTime"],
            ["2009-03-01T12:34:56.007", "localDateTime"],
            ["12:34:56Z", "time"],
            ["12:34:56.250Z", "time"],
            ["12:34:56", "localTime"],
            ["00:00:00", "localTime"],
            ["0050-01-02", "date"],
            ["-0044-03-15", "date"],
            ["10000-01-01", "date"],
        ] as const) {
            const parsed = parseTemporal(text, kind);
            expect(parsed.text, text).toBeNull();
            expect(formatTemporal(parsed.value, kind), text).toBe(text);
        }
    });

    it("handles two-digit years without the Date.UTC 1900 rule", () => {
        expect(parseTemporal("0050-01-02", "date").value).toBe(new Date("0050-01-02T00:00:00Z").getTime());
    });

    it("maps time-of-day to milliseconds since midnight, offsets applied for time", () => {
        expect(parseTemporal("01:02:03", "localTime").value).toBe(3723000);
        expect(parseTemporal("01:02:03.5", "localTime")).toEqual({ value: 3723500, text: "01:02:03.5" });
        expect(parseTemporal("01:00:00+02:00", "time")).toEqual({ value: 23 * 3600000, text: "01:00:00+02:00" });
        expect(parseTemporal("23:00:00-02:00", "time").value).toBe(1 * 3600000);
        expect(parseTemporal("12:00:00+0130", "time").value).toBe((10 * 60 + 30) * 60000);
        expect(parseTemporal("12:00:00+01", "time").value).toBe(11 * 3600000);
        expect(formatTemporal(3723000, "time")).toBe("01:02:03Z");
        expect(formatTemporal(-1000, "localTime")).toBe("23:59:59");
    });

    it("accepts a space separator and lower-case z", () => {
        expect(parseTemporal("2009-03-01 12:00:00z", "dateTime").value).toBe(Date.UTC(2009, 2, 1, 12));
    });

    it("rejects non-ISO and out-of-range text with E_COLUMN_TYPE", () => {
        expect(codeOf(() => parseTemporal("March 1, 2009", "date"))).toBe("E_COLUMN_TYPE");
        expect(codeOf(() => parseTemporal("2009-13-01", "date"))).toBe("E_COLUMN_TYPE");
        expect(codeOf(() => parseTemporal("2009-02-30", "date"))).toBe("E_COLUMN_TYPE");
        expect(codeOf(() => parseTemporal("2009-03-01T25:00:00Z", "dateTime"))).toBe("E_COLUMN_TYPE");
        expect(codeOf(() => parseTemporal("25:00:00", "localTime"))).toBe("E_COLUMN_TYPE");
        expect(codeOf(() => parseTemporal("12:00", "localTime"))).toBeNull();
        expect(codeOf(() => parseTemporal("", "date"))).toBe("E_COLUMN_TYPE");
        expect(codeOf(() => parseTemporal("2009-03-01", "weird" as "date"))).toBe("E_COLUMN_TYPE");
        expect(codeOf(() => formatTemporal(0, "weird" as "date"))).toBe("E_COLUMN_TYPE");
    });
});

describe("parseTimeText / formatTimeValue (GEXF time bounds)", () => {
    it("reads numbers under integer / double and keeps non-canonical text", () => {
        expect(parseTimeText("5", "integer")).toEqual({ value: 5, text: null });
        expect(parseTimeText("5.0", "double")).toEqual({ value: 5, text: "5.0" });
        expect(parseTimeText(" 1e3 ", "double")).toEqual({ value: 1000, text: "1e3" });
        expect(codeOf(() => parseTimeText("2009", "integer"))).toBeNull();
        expect(codeOf(() => parseTimeText("2009-03-01", "integer"))).toBe("E_COLUMN_TYPE");
    });

    it("reads ISO text under date / dateTime", () => {
        expect(parseTimeText("2009-03-01", "date").value).toBe(Date.UTC(2009, 2, 1));
        expect(parseTimeText("2009-03-01T00:00:00Z", "dateTime").value).toBe(Date.UTC(2009, 2, 1));
    });

    it("guesses for a file without a timeformat: numbers, then dateTime", () => {
        expect(parseTimeText("3.5", null)).toEqual({ value: 3.5, text: null });
        expect(parseTimeText("2009-03-01", null).value).toBe(Date.UTC(2009, 2, 1));
        expect(codeOf(() => parseTimeText("soon", null))).toBe("E_COLUMN_TYPE");
        expect(codeOf(() => parseTimeText("1", "weird" as "integer"))).toBe("E_COLUMN_TYPE");
    });

    it("formats back per timeformat", () => {
        expect(formatTimeValue(5, "integer")).toBe("5");
        expect(formatTimeValue(5.5, "double")).toBe("5.5");
        expect(formatTimeValue(Infinity, null)).toBe("Infinity");
        expect(formatTimeValue(Date.UTC(2009, 2, 1), "date")).toBe("2009-03-01");
        expect(formatTimeValue(Date.UTC(2009, 2, 1, 1), "dateTime")).toBe("2009-03-01T01:00:00Z");
        expect(codeOf(() => formatTimeValue(1, "weird" as "integer"))).toBe("E_COLUMN_TYPE");
    });
});

describe("timeTextCompanion", () => {
    it("declares <column>.text with role timeText and extra.for", () => {
        expect(timeTextCompanion("start")).toEqual({
            name: "start.text",
            dtype: "string",
            role: "timeText",
            nullable: true,
            extra: { for: "start" },
        });
        expect(TIME_TEXT_ROLE).toBe("timeText");
        expect(TIME_TEXT_SUFFIX).toBe(".text");
    });
});
