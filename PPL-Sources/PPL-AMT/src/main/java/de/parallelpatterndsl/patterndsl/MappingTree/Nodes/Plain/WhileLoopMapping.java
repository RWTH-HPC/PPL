package de.parallelpatterndsl.patterndsl.MappingTree.Nodes.Plain;

import de.parallelpatterndsl.patterndsl.MappingTree.Nodes.MappingNode;
import de.parallelpatterndsl.patterndsl.MappingTree.Visitor.AMTVisitor;
import de.parallelpatterndsl.patterndsl.abstractPatternTree.DataElements.Data;
import de.parallelpatterndsl.patterndsl.abstractPatternTree.Nodes.PatternNode;
import de.parallelpatterndsl.patterndsl.hardwareDescription.Node;

import java.util.HashMap;
import java.util.Optional;

/**
 * Definition of a while-loop in the abstract mapping tree
 */
public class WhileLoopMapping extends SerialNodeMapping {

    /**
     * Stores the condition of the while loop.
     */
    private ComplexExpressionMapping condition;

    public WhileLoopMapping(Optional<MappingNode> parent, HashMap<String, Data> variableTable, PatternNode node, Node target) {
        super(parent, variableTable, node, target);
    }

    public ComplexExpressionMapping getCondition() {
        return condition;
    }

    public void setCondition(ComplexExpressionMapping condition) {
        this.condition = condition;
    }

    /**
     * Visitor accept function.
     */
    public void accept(AMTVisitor visitor) {
        visitor.handle(this);
    }
}
