package de.parallelpatterndsl.patterndsl.abstractPatternTree.StructurePrinter;

import java.util.HashMap;

/**
 * Definition of all printable tree structures.
 */
public enum TreeDefinition {

    COMPLETE,
    CALL,
    PATTERN_NESTING;

    private static final HashMap<TreeDefinition,String> treeNames;
    static {
        treeNames = new HashMap<>();
        treeNames.put(COMPLETE,"_Complete_Tree");
        treeNames.put(CALL,"_Call_Tree");
        treeNames.put(PATTERN_NESTING,"_Pattern_Nesting_Tree");
    }

    public static HashMap<TreeDefinition, String> getTreeNames() {
        return treeNames;
    }
}
