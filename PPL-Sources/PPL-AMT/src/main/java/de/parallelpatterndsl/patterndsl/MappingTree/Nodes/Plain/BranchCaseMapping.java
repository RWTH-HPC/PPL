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
 * Defines a single path in a branch in an abstract mapping tree.
 */
public class BranchCaseMapping extends MappingNode {

    /**
     * Stores the condition of the branch.
     */
    private Optional<ComplexExpressionMapping> condition;

    public BranchCaseMapping(Optional<MappingNode> parent, HashMap<String, Data> variableTable, PatternNode node) {
        super(parent, variableTable, node);
    }

    public ComplexExpressionMapping getCondition() {
        return condition.get();
    }

    public boolean hasCondition() {
        return condition.isPresent();
    }

    public void setCondition(Optional<ComplexExpressionMapping> condition) {
        this.condition = condition;
    }



    /**
     * Visitor accept function.
     */
    public void accept(AMTVisitor visitor) {
        visitor.handle(this);
    }
}
