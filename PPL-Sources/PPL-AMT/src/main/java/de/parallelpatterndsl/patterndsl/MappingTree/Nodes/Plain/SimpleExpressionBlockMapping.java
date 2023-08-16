package de.parallelpatterndsl.patterndsl.MappingTree.Nodes.Plain;

import de.parallelpatterndsl.patterndsl.MappingTree.Nodes.MappingNode;
import de.parallelpatterndsl.patterndsl.MappingTree.Visitor.AMTVisitor;
import de.parallelpatterndsl.patterndsl.abstractPatternTree.DataElements.Data;
import de.parallelpatterndsl.patterndsl.abstractPatternTree.Nodes.Plain.SimpleExpressionBlockNode;
import de.parallelpatterndsl.patterndsl.expressions.IRLExpression;
import de.parallelpatterndsl.patterndsl.hardwareDescription.Node;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.Optional;

/**
 * A set of expression without function calls within the abstract mapping tree
 */
public class SimpleExpressionBlockMapping extends SerialNodeMapping {

    /**
     * Stores an ordered list of simple sequential operations
     * It starts after the last complex pattern and ends before the next complex pattern
     */
    private ArrayList<IRLExpression> expressionList;

    public SimpleExpressionBlockMapping(Optional<MappingNode> parent, HashMap<String, Data> variableTable, SimpleExpressionBlockNode aptNode, Node target) {
        super(parent, variableTable, aptNode, target);
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
