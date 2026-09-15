/**
 * Nested XML inside a GraphML `<data>` element (yFiles / yEd `y:ShapeNode`, `y:PolyLineEdge`,
 * ...) as a JSON tree, the value of a `json` column with `origin.namespace: "yfiles"` (design
 * section 8.5: structure preserved, not byte-exact). The shape follows the fast-xml-parser
 * convention graphty-element's parser already navigates: an element is an object whose
 * `@_<name>` keys are its attributes, whose `#text` key is its non-whitespace character data
 * when it also has attributes or children, and whose other keys are its child elements (an
 * array when a name repeats); an element with neither attributes nor children is its text
 * (an empty string for an empty element).
 *
 * The builder consumes the tokenizer events of one subtree; the writer produces the XML text of
 * a tree again, so a yFiles column re-exports as the same nested markup.
 */

import { GraphFormatError } from "@graphty/graph-format";

import { escapeXmlAttribute, escapeXmlText } from "../../common/escape.js";
import { isWhitespace, isXmlName } from "../../common/xml.js";

/** The prefix of an attribute key in a tree object. */
const ATTRIBUTE_PREFIX = "@_";

/** The key of an element's character data when it also has attributes or children. */
const TEXT_KEY = "#text";

/** A tree object: attributes, text and child elements by name. */
type XmlTreeObject = Record<string, unknown>;

/** A frame of the builder: one open element. */
interface Frame {
    readonly name: string;
    readonly node: XmlTreeObject;
    hasAttrs: boolean;
    hasChildren: boolean;
    text: string;
}

/**
 * Builds the tree of one subtree from start / end / text events. The outermost frame is the
 * `<data>` element itself; `finish()` returns its content: the object of its children, or its
 * text when it has none.
 */
export class XmlTreeBuilder {
    private readonly frames: Frame[] = [];

    private readonly root: Frame;

    /** Create a builder whose implicit root is the containing element. */
    constructor() {
        this.root = { name: "", node: {}, hasAttrs: false, hasChildren: false, text: "" };
    }

    /**
     * Whether the builder is inside a child element (as opposed to at the root level).
     * @returns true while at least one child element is open
     */
    get open(): boolean {
        return this.frames.length > 0;
    }

    /**
     * An element starts.
     * @param name - the element name as written
     * @param attrs - its attributes
     */
    start(name: string, attrs: ReadonlyMap<string, string>): void {
        const node: XmlTreeObject = {};
        let hasAttrs = false;
        for (const [key, value] of attrs) {
            node[ATTRIBUTE_PREFIX + key] = value;
            hasAttrs = true;
        }
        this.frames.push({ name, node, hasAttrs, hasChildren: false, text: "" });
    }

    /**
     * Character data inside the current element.
     * @param text - the text
     */
    text(text: string): void {
        const frame = this.frames.length > 0 ? this.frames[this.frames.length - 1] : this.root;
        frame.text += text;
    }

    /** The current element ends; its value is attached to its parent. */
    end(): void {
        const frame = this.frames.pop();
        if (frame === undefined) {
            throw new GraphFormatError("E_COLUMN_TYPE", "XmlTreeBuilder.end() without a matching start()", {
                reason: "tree builder underflow",
            });
        }
        const parent = this.frames.length > 0 ? this.frames[this.frames.length - 1] : this.root;
        attachChild(parent, frame.name, valueOf(frame));
    }

    /**
     * The content of the containing element: its text when it has no child elements, otherwise
     * the object of its children (with `#text` for non-whitespace mixed content).
     * @returns the value to store in the json column
     */
    finish(): unknown {
        if (this.frames.length > 0) {
            throw new GraphFormatError("E_COLUMN_TYPE", "XmlTreeBuilder.finish() with open elements", {
                reason: "tree builder open",
            });
        }
        const { root } = this;
        if (!root.hasChildren) {
            return root.text;
        }
        if (!isWhitespace(root.text)) {
            root.node[TEXT_KEY] = root.text;
        }
        return root.node;
    }
}

/**
 * The value of a finished element: its text when it has neither attributes nor children,
 * else its node with `#text` set to non-whitespace text.
 * @param frame - the finished frame
 * @returns the value
 */
function valueOf(frame: Frame): unknown {
    if (!frame.hasAttrs && !frame.hasChildren) {
        return frame.text;
    }
    if (!isWhitespace(frame.text)) {
        frame.node[TEXT_KEY] = frame.text;
    }
    return frame.node;
}

/**
 * Attach a child value under a name, turning a repeated name into an array.
 * @param parent - the parent frame
 * @param name - the child element name
 * @param value - the child value
 */
function attachChild(parent: Frame, name: string, value: unknown): void {
    parent.hasChildren = true;
    const existing = parent.node[name];
    if (existing === undefined) {
        parent.node[name] = value;
    } else if (Array.isArray(existing)) {
        existing.push(value);
    } else {
        parent.node[name] = [existing, value];
    }
}

/**
 * Whether a value is a tree the writer can serialise: a string, number, boolean or null (text),
 * an array of such trees, or an object whose keys are `#text`, `@_<Name>` or `<Name>` with tree
 * values.
 * @param value - the value
 * @returns true when writeXmlTree() accepts it
 */
export function isXmlTree(value: unknown): boolean {
    return treeProblem(value) === null;
}

/**
 * Why a value is not a serialisable tree.
 * @param value - the value
 * @returns a message, or null when the value is a tree
 */
export function treeProblem(value: unknown): string | null {
    if (value === null || typeof value === "string" || typeof value === "number" || typeof value === "boolean") {
        return null;
    }
    if (Array.isArray(value)) {
        for (const item of value) {
            if (Array.isArray(item)) {
                return "nested arrays cannot be written as elements";
            }
            const problem = treeProblem(item);
            if (problem !== null) {
                return problem;
            }
        }
        return null;
    }
    if (typeof value !== "object") {
        return `a ${typeof value} cannot be written as XML`;
    }
    for (const [key, child] of Object.entries(value)) {
        if (key === TEXT_KEY) {
            if (child !== null && typeof child === "object") {
                return "#text must be a scalar";
            }
            continue;
        }
        if (key.startsWith(ATTRIBUTE_PREFIX)) {
            const name = key.slice(ATTRIBUTE_PREFIX.length);
            if (!isXmlName(name)) {
                return `"${name}" is not an XML attribute name`;
            }
            if (child !== null && typeof child === "object") {
                return `attribute ${name} must be a scalar`;
            }
            continue;
        }
        if (!isXmlName(key)) {
            return `"${key}" is not an XML element name`;
        }
        const problem = treeProblem(child);
        if (problem !== null) {
            return problem;
        }
    }
    return null;
}

/**
 * Write the content of a tree as XML text: the children (and mixed text) of the containing
 * element. Leaf elements are written inline so their text stays exact; elements with children
 * are broken over lines and indented.
 * @param value - the tree (the content of a `<data>` element)
 * @param indent - the indentation of the containing element's children
 * @param out - receives the text parts
 */
export function writeXmlTree(value: unknown, indent: string, out: string[]): void {
    const problem = treeProblem(value);
    if (problem !== null) {
        throw new GraphFormatError("E_COLUMN_TYPE", `a yfiles value cannot be written as XML: ${problem}`, {
            reason: "xml tree",
        });
    }
    if (value === null || typeof value !== "object" || Array.isArray(value)) {
        out.push(escapeXmlText(scalarText(value)));
        return;
    }
    const object = value as XmlTreeObject;
    let text: string | null = null;
    for (const [key, child] of Object.entries(object)) {
        if (key === TEXT_KEY) {
            text = scalarText(child);
        } else if (!key.startsWith(ATTRIBUTE_PREFIX)) {
            writeElements(key, child, indent, out);
        }
    }
    if (text !== null) {
        out.push(escapeXmlText(text));
    }
}

/**
 * Write one element (or, for an array, one element per item).
 * @param name - the element name
 * @param value - the element value
 * @param indent - the indentation of this element
 * @param out - receives the text parts
 */
function writeElements(name: string, value: unknown, indent: string, out: string[]): void {
    if (Array.isArray(value)) {
        for (const item of value) {
            writeElements(name, item, indent, out);
        }
        return;
    }
    if (value === null || typeof value !== "object") {
        const text = scalarText(value);
        out.push(text.length === 0 ? `\n${indent}<${name}/>` : `\n${indent}<${name}>${escapeXmlText(text)}</${name}>`);
        return;
    }
    const object = value as XmlTreeObject;
    let attrs = "";
    let text: string | null = null;
    let hasChildren = false;
    for (const [key, child] of Object.entries(object)) {
        if (key === TEXT_KEY) {
            text = scalarText(child);
        } else if (key.startsWith(ATTRIBUTE_PREFIX)) {
            attrs += ` ${key.slice(ATTRIBUTE_PREFIX.length)}="${escapeXmlAttribute(scalarText(child))}"`;
        } else {
            hasChildren = true;
        }
    }
    if (!hasChildren) {
        out.push(
            text === null || text.length === 0
                ? `\n${indent}<${name}${attrs}/>`
                : `\n${indent}<${name}${attrs}>${escapeXmlText(text)}</${name}>`,
        );
        return;
    }
    out.push(`\n${indent}<${name}${attrs}>`);
    const inner = `${indent}  `;
    for (const [key, child] of Object.entries(object)) {
        if (key !== TEXT_KEY && !key.startsWith(ATTRIBUTE_PREFIX)) {
            writeElements(key, child, inner, out);
        }
    }
    if (text !== null) {
        out.push(escapeXmlText(text));
    }
    out.push(`\n${indent}</${name}>`);
}

/**
 * The text of a scalar tree value.
 * @param value - a string, number, boolean or null
 * @returns the text; an empty string for null
 */
function scalarText(value: unknown): string {
    switch (typeof value) {
        case "string":
            return value;
        case "number":
        case "boolean":
        case "bigint":
            return String(value);
        default:
            return "";
    }
}
