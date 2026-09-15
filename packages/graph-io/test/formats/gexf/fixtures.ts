/**
 * Hand-written GEXF documents for the importer / exporter tests: a 1.3 dynamic graph exercising
 * every construct the importer maps (typed attributes with defaults and options, lists, temporal
 * types with companions, viz, containment, lifetimes, dynamic values, mixed direction, mutual
 * edges, parallel edges, self-loops, the weight override), a 1.2 document with open intervals and
 * pipe lists, and small documents for the option and error paths.
 */

/** A non-ASCII label built at runtime so this source file stays plain ASCII. */
export const ACCENTED = `Orl${String.fromCharCode(0xe9)}ans`;

/** A 1.3 dynamic graph with every construct. */
export const DYNAMIC_1_3 = `<?xml version="1.0" encoding="UTF-8"?>
<gexf xmlns="http://gexf.net/1.3" xmlns:viz="http://gexf.net/1.3/viz" version="1.3">
  <meta lastmodifieddate="2024-05-01">
    <creator>graph-io tests</creator>
    <description>a &amp; b &lt; c &#233; &#x41;</description>
    <keywords>alpha, beta</keywords>
  </meta>
  <graph defaultedgetype="directed" mode="dynamic" timeformat="date" timerepresentation="interval" idtype="string" start="2020-01-01" end="2021-01-01">
    <attributes class="node" mode="static">
      <attribute id="0" title="name" type="string"><default>anon</default></attribute>
      <attribute id="1" title="age" type="integer"><default>0</default></attribute>
      <attribute id="2" title="score" type="double"/>
      <attribute id="3" title="active" type="boolean"><default>true</default></attribute>
      <attribute id="4" title="tags" type="liststring"/>
      <attribute id="5" title="born" type="date"/>
      <attribute id="6" title="big" type="long"/>
      <attribute id="7" title="cat" type="string"><options>[a, b, c]</options><default>b</default></attribute>
      <attribute id="8" title="label" type="string"/>
      <attribute id="9" title="ratio" type="float"/>
      <attribute id="10" title="nums" type="listinteger"/>
      <attribute id="11" title="url" type="anyURI"/>
    </attributes>
    <attributes class="node" mode="dynamic">
      <attribute id="price" title="price" type="double"/>
    </attributes>
    <attributes class="edge" mode="static">
      <attribute id="e0" title="rel" type="string"/>
    </attributes>
    <attributes class="edge" mode="dynamic">
      <attribute id="w" title="weight" type="double"/>
    </attributes>
    <nodes count="5">
      <node id="a" label="Alice" start="2020-01-01" end="2020-12-31">
        <attvalues>
          <attvalue for="0" value="Alice &amp; co"/>
          <attvalue for="1" value="30"/>
          <attvalue for="2" value="1.5"/>
          <attvalue for="3" value="false"/>
          <attvalue for="4" value="[x, 'y, z', w]"/>
          <attvalue for="5" value="1990-05-06"/>
          <attvalue for="6" value="9007199254740993"/>
          <attvalue for="7" value="c"/>
          <attvalue for="8" value="renamed"/>
          <attvalue for="9" value="0.25"/>
          <attvalue for="10" value="[1, 2, 3]"/>
          <attvalue for="11" value="http://example.com/?q=1&amp;r=2"/>
          <attvalue for="price" value="10" start="2020-01-01" end="2020-06-30"/>
          <attvalue for="price" value="12.5" start="2020-07-01"/>
          <attvalue for="price" value="11"/>
        </attvalues>
        <viz:color r="255" g="0" b="0" a="0.5"/>
        <viz:position x="1.5" y="-2" z="3"/>
        <viz:size value="2.5"/>
        <viz:shape value="square"/>
      </node>
      <node id="b" label="${ACCENTED}" pid="a" timestamps="&lt;[2020-02-01, 2020-03-01]&gt;">
        <attvalues>
          <attvalue for="5" value="2001-02-03T04:05:06+02:00"/>
        </attvalues>
        <viz:color hex="#00FF00"/>
        <viz:position x="0" y="0"/>
        <spells>
          <spell start="2020-01-01" end="2020-02-01"/>
          <spell start="2020-03-01"/>
        </spells>
      </node>
      <node id="c" label="Carol">
        <parents>
          <parent for="a"/>
          <parent for="b"/>
        </parents>
        <nodes>
          <node id="d" label="Dan"/>
        </nodes>
      </node>
      <node id="1" label="One"/>
    </nodes>
    <edges count="6">
      <edge id="e1" source="a" target="b" weight="2.5" label="knows" start="2020-01-01">
        <attvalues>
          <attvalue for="e0" value="friend"/>
          <attvalue for="w" value="3.5" start="2020-06-01" end="2020-07-01"/>
        </attvalues>
        <viz:color r="0" g="0" b="255"/>
        <viz:thickness value="3"/>
        <viz:shape value="dotted"/>
      </edge>
      <edge id="e2" source="b" target="c" type="undirected"/>
      <edge id="e3" source="c" target="a" type="mutual" weight="1"/>
      <edge id="e4" source="a" target="a" type="undirected"/>
      <edge id="e5" source="a" target="b" kind="second"/>
      <edge id="e6" source="d" target="1">
        <attvalues>
          <attvalue for="w" value="7"/>
        </attvalues>
      </edge>
    </edges>
  </graph>
</gexf>
`;

/** A 1.2 document: integer times, open intervals, pipe lists, dynamic viz, a mutual default. */
export const OPEN_1_2 = `<?xml version="1.0" encoding="UTF-8"?>
<gexf xmlns="http://www.gexf.net/1.2draft" xmlns:viz="http://www.gexf.net/1.2draft/viz" version="1.2">
  <graph defaultedgetype="mutual" mode="dynamic" timeformat="integer">
    <attributes class="node" mode="static">
      <attribute id="0" title="tags" type="liststring"/>
      <attribute id="1" title="flag" type="boolean"/>
    </attributes>
    <attributes class="node" mode="dynamic">
      <attribute id="2" title="level" type="integer"/>
    </attributes>
    <nodes>
      <node id="1" label="one" startopen="1" end="5">
        <attvalues>
          <attvalue for="0" value="x|y|z"/>
          <attvalue for="1" value="1"/>
          <attvalue for="2" value="3" start="1" endopen="2"/>
          <attvalue for="2" value="4" start="2"/>
        </attvalues>
        <viz:color r="1" g="2" b="3" start="1" end="2"/>
      </node>
      <node id="2" label="two" endopen="9">
        <spells>
          <spell startopen="1" end="2"/>
        </spells>
      </node>
    </nodes>
    <edges>
      <edge id="0" source="1" target="2" weight="0.1"/>
      <edge id="1" source="2" target="1" type="directed" start="3" endopen="4"/>
    </edges>
  </graph>
</gexf>
`;

/** A document that declares no direction, no ids type and no meta. */
export const BARE = `<gexf version="1.2">
  <graph>
    <nodes>
      <node id="x"/>
      <node id="y"/>
      <node id="01"/>
    </nodes>
    <edges>
      <edge source="x" target="y"/>
      <edge source="y" target="01" weight="2"/>
    </edges>
  </graph>
</gexf>
`;

/** A mixed-direction 1.3 document whose header says undirected. */
export const MIXED_UNDIRECTED_HEADER = `<gexf xmlns="http://gexf.net/1.3" version="1.3">
  <graph defaultedgetype="undirected">
    <nodes>
      <node id="p"/>
      <node id="q"/>
      <node id="r"/>
    </nodes>
    <edges>
      <edge id="u1" source="p" target="q"/>
      <edge id="d1" source="q" target="r" type="directed"/>
      <edge id="m1" source="r" target="p" type="mutual"/>
    </edges>
  </graph>
</gexf>
`;

/** A document whose values break the declared types and reference unknown things. */
export const SLOPPY = `<gexf xmlns="http://gexf.net/1.3" version="1.3">
  <graph defaultedgetype="directed" mode="wrong" timeformat="weird">
    <attributes class="node" mode="static">
      <attribute id="0" title="n" type="integer"><default>notanumber</default></attribute>
      <attribute id="0" title="dup" type="string"/>
      <attribute id="1" title="t" type="mystery"/>
      <attribute id="2" title="u"/>
      <attribute title="noid" type="string"/>
    </attributes>
    <attributes>
      <attribute id="9" title="orphan" type="string"/>
    </attributes>
    <nodes>
      <node id="a">
        <attvalues>
          <attvalue for="0" value="12"/>
          <attvalue for="0" value="twelve"/>
          <attvalue for="unknown" value="1"/>
          <attvalue for="1" value="kept as text"/>
          <attvalue value="no for"/>
          <attvalue for="2"/>
        </attvalues>
        <viz:position x="nope" y="1"/>
      </node>
      <node id="a"/>
      <node id="b" pid="ghost"/>
      <node id="c"/>
    </nodes>
    <edges>
      <edge id="ok" source="a" target="b"/>
      <edge id="badtype" source="a" target="b" type="sideways"/>
      <edge id="badweight" source="a" target="b" weight="heavy"/>
      <edge id="noend" source="a"/>
      <edge id="unknown" source="a" target="zzz"/>
    </edges>
  </graph>
</gexf>
`;
