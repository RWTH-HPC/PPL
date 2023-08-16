package de.parallelpatterndsl.patterndsl.MappingTree.Nodes.Plain;

import de.parallelpatterndsl.patterndsl.MappingTree.Nodes.MappingNode;
import de.parallelpatterndsl.patterndsl.MappingTree.Visitor.AMTVisitor;
import de.parallelpatterndsl.patterndsl.abstractPatternTree.DataElements.Data;
import de.parallelpatterndsl.patterndsl.abstractPatternTree.Nodes.Plain.LoopSkipNode;
import de.parallelpatterndsl.patterndsl.hardwareDescription.Node;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.Optional;

public class LoopSkipMapping extends SerialNodeMapping {

    /**
     * True, iff the key word is break, else the key word is continue.
     */
    private boolean isBreak;

    public LoopSkipMapping(Optional<MappingNode> parent, HashMap<String, Data> variableTable, LoopSkipNode node, Node target) {
        super(parent, variableTable, node, target);
        this.isBreak = node.isBreak();
        this.setChildren(new ArrayList<>());
    }

    public boolean isBreak() {
        return isBreak;
    }


    /**
     * Visitor accept function.
     */
    public void accept(AMTVisitor visitor) {
        visitor.handle(this);
    }
}
