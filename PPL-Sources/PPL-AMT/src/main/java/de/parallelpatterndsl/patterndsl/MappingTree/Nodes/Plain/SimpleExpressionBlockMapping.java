package de.parallelpatterndsl.patterndsl.MappingTree.Nodes.Plain;

import de.parallelpatterndsl.patterndsl.MappingTree.AbstractMappingTree;
import de.parallelpatterndsl.patterndsl.MappingTree.Nodes.DataControl.DataPlacement;
import de.parallelpatterndsl.patterndsl.MappingTree.Nodes.DataControl.EndPoint;
import de.parallelpatterndsl.patterndsl.MappingTree.Nodes.MappingNode;
import de.parallelpatterndsl.patterndsl.MappingTree.Visitor.AMTVisitor;
import de.parallelpatterndsl.patterndsl.abstractPatternTree.DataElements.ArrayData;
import de.parallelpatterndsl.patterndsl.abstractPatternTree.DataElements.Data;
import de.parallelpatterndsl.patterndsl.abstractPatternTree.Nodes.Plain.SimpleExpressionBlockNode;
import de.parallelpatterndsl.patterndsl.expressions.IRLExpression;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.HashSet;
import java.util.Optional;

/**
 * A set of expression without function calls within the abstract mapping tree
 */
public class SimpleExpressionBlockMapping extends MappingNode {

    /**
     * Stores an ordered list of simple sequential operations
     * It starts after the last complex pattern and ends before the next complex pattern
     */
    private ArrayList<IRLExpression> expressionList;

    public SimpleExpressionBlockMapping(Optional<MappingNode> parent, HashMap<String, Data> variableTable, SimpleExpressionBlockNode aptNode) {
        super(parent, variableTable, aptNode);
        expressionList = aptNode.getExpressionList();
        children = new ArrayList<>();
    }



    public ArrayList<IRLExpression> getExpressionList() {
        return expressionList;
    }

    /**
     * Visitor accept function.
     */
    public void accept(AMTVisitor visitor) {
        visitor.handle(this);
    }
}
