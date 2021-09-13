package de.parallelpatterndsl.patterndsl.MappingTree.Nodes.Plain;

import de.parallelpatterndsl.patterndsl.MappingTree.AbstractMappingTree;
import de.parallelpatterndsl.patterndsl.MappingTree.Nodes.DataControl.DataPlacement;
import de.parallelpatterndsl.patterndsl.MappingTree.Nodes.DataControl.EndPoint;
import de.parallelpatterndsl.patterndsl.MappingTree.Nodes.MappingNode;
import de.parallelpatterndsl.patterndsl.MappingTree.Visitor.AMTVisitor;
import de.parallelpatterndsl.patterndsl.abstractPatternTree.DataElements.ArrayData;
import de.parallelpatterndsl.patterndsl.abstractPatternTree.DataElements.Data;
import de.parallelpatterndsl.patterndsl.abstractPatternTree.Nodes.PatternNode;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.HashSet;
import java.util.Optional;

/**
 * Definition of a while-loop in the abstract mapping tree
 */
public class WhileLoopMapping extends MappingNode {

    /**
     * Stores the condition of the while loop.
     */
    private ComplexExpressionMapping condition;

    public WhileLoopMapping(Optional<MappingNode> parent, HashMap<String, Data> variableTable, PatternNode node) {
        super(parent, variableTable, node);
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
