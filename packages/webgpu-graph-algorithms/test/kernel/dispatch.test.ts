/**
 * Dispatch planning (spec 5.2, contract 3.9; 5.5 row dispatch.test.ts): the 16,776,960 rule with wg 256 (NOT 2^24),
 * the empty plan, the 2D form, E_TOO_LARGE above 65,535^2 groups, the P1-P3 E_UNSUPPORTED stubs, and the same plans
 * from every capability table (no dependence on subgroup sizes). Pure: no device.
 */

import { MAX_1D_ITEMS, MAX_WORKGROUPS_PER_DIM } from "../../src/constants.js";
import { isWebGpuGraphError, type WebGpuGraphError } from "../../src/errors.js";
import {
    type DispatchPlan,
    groupsOf,
    plan1d,
    plan2d,
    planGridStride,
    planIndirect,
} from "../../src/kernel/dispatch.js";
import { CAPS_SPEC_DEFAULT, CAPS_TABLES, fakeCaps } from "../helpers/caps-tables.js";

const LIMIT = MAX_WORKGROUPS_PER_DIM;
const WG = 256;

function catchError(fn: () => unknown): WebGpuGraphError {
    try {
        fn();
    } catch (err) {
        if (isWebGpuGraphError(err)) {
            return err;
        }
        throw err;
    }
    throw new Error("expected a WebGpuGraphError");
}

function plan(x: number, y: number, items: number): DispatchPlan {
    return { x, y, z: 1, items, stride: null };
}

describe("plan1d", () => {
    it("MAX_1D_ITEMS is 65,535 x 256 = 16,776,960 (design 10.6), not 2^24", () => {
        expect(MAX_1D_ITEMS).toBe(16_776_960);
        expect(LIMIT * WG).toBe(16_776_960);
        expect(2 ** 24).toBe(16_777_216);
    });

    it("16,776,960 items are 1D and 16,776,961 are 2D with wg 256", () => {
        expect(plan1d(16_776_960, WG, CAPS_SPEC_DEFAULT)).toEqual(plan(65_535, 1, 16_776_960));
        expect(plan1d(16_776_961, WG, CAPS_SPEC_DEFAULT)).toEqual(plan(65_535, 2, 16_776_961));
        // the round 2^24 is already 2D: 65,536 groups
        expect(plan1d(2 ** 24, WG, CAPS_SPEC_DEFAULT)).toEqual(plan(65_535, 2, 16_777_216));
    });

    it("counts groups with ceil(items / wg)", () => {
        expect(plan1d(1, WG, CAPS_SPEC_DEFAULT)).toEqual(plan(1, 1, 1));
        expect(plan1d(255, WG, CAPS_SPEC_DEFAULT)).toEqual(plan(1, 1, 255));
        expect(plan1d(256, WG, CAPS_SPEC_DEFAULT)).toEqual(plan(1, 1, 256));
        expect(plan1d(257, WG, CAPS_SPEC_DEFAULT)).toEqual(plan(2, 1, 257));
        expect(plan1d(65_534 * WG, WG, CAPS_SPEC_DEFAULT)).toEqual(plan(65_534, 1, 65_534 * WG));
        expect(plan1d(65_535 * 64, 64, CAPS_SPEC_DEFAULT)).toEqual(plan(65_535, 1, 65_535 * 64));
        expect(plan1d(65_535 * 64 + 1, 64, CAPS_SPEC_DEFAULT)).toEqual(plan(65_535, 2, 65_535 * 64 + 1));
    });

    it("plan1d(0) is the empty plan { x: 0, y: 1 } (spec 5.6)", () => {
        expect(plan1d(0, WG, CAPS_SPEC_DEFAULT)).toEqual(plan(0, 1, 0));
        expect(groupsOf(plan1d(0, WG, CAPS_SPEC_DEFAULT))).toBe(0);
    });

    it("the largest plannable count is 65,535^2 groups; one more group is E_TOO_LARGE", () => {
        const maxItems = LIMIT * LIMIT * WG; // 1,099,478,073,600
        expect(maxItems).toBe(1_099_478_073_600);
        expect(plan1d(maxItems, WG, CAPS_SPEC_DEFAULT)).toEqual(plan(65_535, 65_535, maxItems));
        const err = catchError(() => plan1d(maxItems + 1, WG, CAPS_SPEC_DEFAULT));
        expect(err.code).toBe("E_TOO_LARGE");
        expect(err.details).toMatchObject({
            needed: LIMIT * LIMIT + 1,
            limit: LIMIT * LIMIT,
            path: "dispatch",
            algorithm: null,
        });
    });

    it("rejects a negative or fractional item count and a workgroup size that is not a power of two", () => {
        for (const items of [-1, 1.5, Number.NaN]) {
            const err = catchError(() => plan1d(items, WG, CAPS_SPEC_DEFAULT));
            expect(err.code).toBe("E_INVALID_ARGUMENT");
            expect(err.details.argument).toBe("items");
        }
        for (const wg of [0, 3, 96, 0.5, -256]) {
            const err = catchError(() => plan1d(10, wg, CAPS_SPEC_DEFAULT));
            expect(err.code).toBe("E_INVALID_ARGUMENT");
            expect(err.details.argument).toBe("wg");
        }
        expect(plan1d(10, 1, CAPS_SPEC_DEFAULT)).toEqual(plan(10, 1, 10));
    });

    it("gives the same plans from every capability table incl. a min != max subgroup table", () => {
        const tables = [
            ...CAPS_TABLES.map((t) => t.caps),
            fakeCaps(CAPS_SPEC_DEFAULT, {}, { subgroupMinSize: 8, subgroupMaxSize: 32 }),
        ];
        expect(CAPS_TABLES.length).toBeGreaterThanOrEqual(4);
        for (const caps of tables) {
            expect(caps.limits.maxComputeWorkgroupsPerDimension).toBe(LIMIT);
            expect(plan1d(16_776_960, WG, caps)).toEqual(plan(65_535, 1, 16_776_960));
            expect(plan1d(16_776_961, WG, caps)).toEqual(plan(65_535, 2, 16_776_961));
            expect(plan1d(0, WG, caps)).toEqual(plan(0, 1, 0));
            expect(plan1d(1000, WG, caps)).toEqual(plan(4, 1, 1000));
        }
    });

    it("honours a faked per-dimension limit below the constant (the device limit is asserted equal at create())", () => {
        const small = fakeCaps(CAPS_SPEC_DEFAULT, { maxComputeWorkgroupsPerDimension: 4 });
        expect(plan1d(4 * WG, WG, small)).toEqual(plan(4, 1, 4 * WG));
        expect(plan1d(4 * WG + 1, WG, small)).toEqual(plan(4, 2, 4 * WG + 1));
        expect(catchError(() => plan1d(17 * WG, WG, small)).code).toBe("E_TOO_LARGE");
    });
});

describe("plan2d", () => {
    it("is plan1d's rule on a group count (spec 5.4): 1D up to 65,535 groups, then x = 65,535, y = ceil(groups / 65,535)", () => {
        expect(plan2d(1, CAPS_SPEC_DEFAULT)).toEqual(plan(1, 1, 1));
        expect(plan2d(1000, CAPS_SPEC_DEFAULT)).toEqual(plan(1000, 1, 1000));
        expect(plan2d(65_535, CAPS_SPEC_DEFAULT)).toEqual(plan(65_535, 1, 65_535));
        expect(plan2d(65_536, CAPS_SPEC_DEFAULT)).toEqual(plan(65_535, 2, 65_536));
        expect(plan2d(LIMIT * LIMIT, CAPS_SPEC_DEFAULT)).toEqual(plan(65_535, 65_535, LIMIT * LIMIT));
        expect(plan2d(0, CAPS_SPEC_DEFAULT)).toEqual(plan(0, 1, 0));
    });

    it("agrees with plan1d on every group count: plan2d(ceil(items / wg)) is plan1d(items, wg) up to the items field", () => {
        for (const items of [1, 255, 256, 257, 16_776_960, 16_776_961, 2 ** 24, LIMIT * LIMIT * WG]) {
            const groups = Math.ceil(items / WG);
            const viaItems = plan1d(items, WG, CAPS_SPEC_DEFAULT);
            expect(plan2d(groups, CAPS_SPEC_DEFAULT)).toEqual({ ...viaItems, items: groups });
        }
        const small = fakeCaps(CAPS_SPEC_DEFAULT, { maxComputeWorkgroupsPerDimension: 4 });
        expect(plan2d(4, small)).toEqual(plan(4, 1, 4));
        expect(plan2d(5, small)).toEqual(plan(4, 2, 5));
        expect(catchError(() => plan2d(17, small)).code).toBe("E_TOO_LARGE");
    });

    it("is E_TOO_LARGE above 65,535^2 groups and E_INVALID_ARGUMENT for a bad count", () => {
        const err = catchError(() => plan2d(LIMIT * LIMIT + 1, CAPS_SPEC_DEFAULT));
        expect(err.code).toBe("E_TOO_LARGE");
        expect(err.details).toMatchObject({
            needed: LIMIT * LIMIT + 1,
            limit: LIMIT * LIMIT,
            path: "dispatch",
            algorithm: null,
        });
        expect(catchError(() => plan2d(-1, CAPS_SPEC_DEFAULT)).code).toBe("E_INVALID_ARGUMENT");
    });
});

describe("groupsOf", () => {
    it("is x * y", () => {
        expect(groupsOf(plan(65_535, 2, 0))).toBe(131_070);
        expect(groupsOf(plan(3, 1, 0))).toBe(3);
        expect(groupsOf(plan1d(16_776_961, WG, CAPS_SPEC_DEFAULT))).toBe(131_070);
    });
});

describe("the P1-P3 stubs", () => {
    it("planGridStride throws E_UNSUPPORTED { feature: 'planGridStride' } (lead f)", () => {
        const err = catchError(() => planGridStride(1000, WG, CAPS_SPEC_DEFAULT));
        expect(err.code).toBe("E_UNSUPPORTED");
        expect(err.details.feature).toBe("planGridStride");
        expect(err.details.option).toBeUndefined();
        expect(catchError(() => planGridStride(1000, WG, CAPS_SPEC_DEFAULT, 16)).details.feature).toBe(
            "planGridStride",
        );
    });

    it("planIndirect throws E_UNSUPPORTED { feature: 'planIndirect' } (lead f)", () => {
        const err = catchError(() => planIndirect(1000, WG, CAPS_SPEC_DEFAULT));
        expect(err.code).toBe("E_UNSUPPORTED");
        expect(err.details.feature).toBe("planIndirect");
        expect(err.details.option).toBeUndefined();
    });
});
