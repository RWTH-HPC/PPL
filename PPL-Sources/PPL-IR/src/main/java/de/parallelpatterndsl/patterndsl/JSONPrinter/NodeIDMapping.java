package de.parallelpatterndsl.patterndsl.JSONPrinter;

import de.parallelpatterndsl.patterndsl.abstractPatternTree.Nodes.PatternNode;

import java.util.HashMap;

public class NodeIDMapping {

    private static HashMap<String, PatternNode> mapping;

    public static void addMapping(String id, PatternNode node) {
        mapping.put(id, node);
    }

    public static PatternNode getMapping(String id) {
        return mapping.get(id);
    }

    static {
        mapping = new HashMap<>();
    }
}
