package de.parallelpatterndsl.patterndsl.MappingTree.Nodes.Plain;

import de.parallelpatterndsl.patterndsl.MappingTree.Nodes.MappingNode;
import de.parallelpatterndsl.patterndsl.MappingTree.Visitor.AMTVisitor;
import de.parallelpatterndsl.patterndsl.abstractPatternTree.DataElements.Data;
import de.parallelpatterndsl.patterndsl.abstractPatternTree.Nodes.PatternNode;
import de.parallelpatterndsl.patterndsl.hardwareDescription.Node;

import java.util.HashMap;
import java.util.Optional;

/**
 * Defines a branch in the abstract mapping tree.
 */
public class BranchMapping extends SerialNodeMapping {
    public BranchMapping(Optional<MappingNode> parent, HashMap<String, Data> variableTable, PatternNode node, Node target) {
        super(parent, variableTable, node, target);
    }



    /**
     * Visitor accept function.
     */
    public void accept(AMTVisitor visitor) {
        visitor.handle(this);
    }
}
