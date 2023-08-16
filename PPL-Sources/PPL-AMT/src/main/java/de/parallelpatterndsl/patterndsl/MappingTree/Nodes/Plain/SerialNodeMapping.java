package de.parallelpatterndsl.patterndsl.MappingTree.Nodes.Plain;

import de.parallelpatterndsl.patterndsl.MappingTree.Nodes.MappingNode;
import de.parallelpatterndsl.patterndsl.abstractPatternTree.DataElements.Data;
import de.parallelpatterndsl.patterndsl.abstractPatternTree.Nodes.PatternNode;
import de.parallelpatterndsl.patterndsl.hardwareDescription.Node;

import java.util.HashMap;
import java.util.Optional;

/**
 * Defines a single path in a branch in an abstract mapping tree.
 */
public abstract class SerialNodeMapping extends MappingNode {

    /**
     * Stores the condition of the branch.
     */
    private Node targetNode;

    public SerialNodeMapping(Optional<MappingNode> parent, HashMap<String, Data> variableTable, PatternNode node, Node targetNode) {
        super(parent, variableTable, node);
        this.targetNode = targetNode;
    }

    public Node getTargetNode() {
        return targetNode;
    }
}
