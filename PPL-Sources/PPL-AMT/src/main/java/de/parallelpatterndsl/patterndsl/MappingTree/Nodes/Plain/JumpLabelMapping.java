package de.parallelpatterndsl.patterndsl.MappingTree.Nodes.Plain;

import de.parallelpatterndsl.patterndsl.MappingTree.Nodes.MappingNode;
import de.parallelpatterndsl.patterndsl.MappingTree.Visitor.AMTVisitor;
import de.parallelpatterndsl.patterndsl.abstractPatternTree.DataElements.Data;
import de.parallelpatterndsl.patterndsl.abstractPatternTree.Nodes.PatternNode;
import de.parallelpatterndsl.patterndsl.hardwareDescription.Node;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.Optional;

public class JumpLabelMapping extends SerialNodeMapping {

    /**
     * Describes the label connecting the jump-Statement and label node.
     */
    private String label;

    public JumpLabelMapping(Optional<MappingNode> parent, HashMap<String, Data> variableTable, PatternNode node, String label, Node target) {
        super(parent, variableTable, node, target);
        this.label = label;
        this.children = new ArrayList<>();
    }

    public String getLabel() {
        return label;
    }



    /**
     * Visitor accept function.
     */
    public void accept(AMTVisitor visitor) {
        visitor.handle(this);
    }
}
