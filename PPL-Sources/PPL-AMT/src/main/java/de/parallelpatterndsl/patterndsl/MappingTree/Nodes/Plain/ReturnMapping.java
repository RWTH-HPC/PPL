package de.parallelpatterndsl.patterndsl.MappingTree.Nodes.Plain;

import de.parallelpatterndsl.patterndsl.MappingTree.AbstractMappingTree;
import de.parallelpatterndsl.patterndsl.MappingTree.Nodes.DataControl.DataPlacement;
import de.parallelpatterndsl.patterndsl.MappingTree.Nodes.DataControl.EndPoint;
import de.parallelpatterndsl.patterndsl.MappingTree.Nodes.MappingNode;
import de.parallelpatterndsl.patterndsl.MappingTree.Visitor.AMTVisitor;
import de.parallelpatterndsl.patterndsl.abstractPatternTree.DataElements.ArrayData;
import de.parallelpatterndsl.patterndsl.abstractPatternTree.DataElements.Data;
import de.parallelpatterndsl.patterndsl.abstractPatternTree.DataElements.FunctionInlineData;
import de.parallelpatterndsl.patterndsl.abstractPatternTree.DataElements.PrimitiveData;
import de.parallelpatterndsl.patterndsl.abstractPatternTree.Nodes.PatternNode;
import de.parallelpatterndsl.patterndsl.expressions.AssignmentExpression;
import de.parallelpatterndsl.patterndsl.expressions.OperationExpression;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.HashSet;
import java.util.Optional;

/**
 * Defines the return statement in abstract mapping trees.
 */
public class ReturnMapping extends MappingNode {

    /**
     * Stores the result of the return statement.
     */
    private ComplexExpressionMapping result;


    public ReturnMapping(Optional<MappingNode> parent, HashMap<String, Data> variableTable, PatternNode node) {
        super(parent, variableTable, node);
    }

    public ComplexExpressionMapping getResult() {
        return result;
    }

    public void setResult(ComplexExpressionMapping result) {
        this.result = result;
    }

    @Override
    public HashSet<DataPlacement> getNecessaryData() {
        return result.getNecessaryData();
    }

    /**
     * Visitor accept function.
     */
    public void accept(AMTVisitor visitor) {
        visitor.handle(this);
    }
}
