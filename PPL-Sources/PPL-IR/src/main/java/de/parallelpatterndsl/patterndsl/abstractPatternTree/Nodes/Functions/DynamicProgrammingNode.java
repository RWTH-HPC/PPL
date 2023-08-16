package de.parallelpatterndsl.patterndsl.abstractPatternTree.Nodes.Functions;

import de.parallelpatterndsl.patterndsl.abstractPatternTree.Nodes.PatternNode;
import de.parallelpatterndsl.patterndsl.abstractPatternTree.Nodes.Plain.ReturnNode;
import de.parallelpatterndsl.patterndsl.abstractPatternTree.Visitor.APTVisitor;
import de.parallelpatterndsl.patterndsl.abstractPatternTree.Visitor.CallCountResetter;
import de.parallelpatterndsl.patterndsl.abstractPatternTree.Visitor.ExtendedShapeAPTVisitor;
import de.parallelpatterndsl.patterndsl.helperLibrary.DeepCopyHelper;

import java.util.ArrayList;

/**
 * Definition of the Dynamic Programming pattern with in the abstract pattern tree.
 */
public class DynamicProgrammingNode extends ParallelNode {

    /**
     * The dimensionality of the underlying data structure.
     */
    private int dimension;

    public DynamicProgrammingNode(String identifier, int dimension) {
        super(identifier);
        this.dimension = dimension;
    }

    public int getDimension() {
        return dimension;
    }

    @Override
    public DynamicProgrammingNode deepCopy() {
        DynamicProgrammingNode result = new DynamicProgrammingNode(getIdentifier(), dimension);

        result.setVariableTable(DeepCopyHelper.currentScope());

        DeepCopyHelper.DataTraceUpdate(result);

        DeepCopyHelper.addScope(getVariableTable());

        result.setReturnElement(DeepCopyHelper.currentScope().get(getReturnElement().getIdentifier()));

        ArrayList<PatternNode> newChildren = new ArrayList<>();
        for (PatternNode node: getChildren()) {
            PatternNode newNode = node.deepCopy();
            newChildren.add(newNode);
            newNode.setParent(result);
        }
        result.setChildren(newChildren);
        DeepCopyHelper.removeScope();

        return result;
    }

    /**
     * Visitor accept function.
     */
    @Override
    public void accept(APTVisitor visitor) {
        visitor.handle(this);
    }

    public void accept(ExtendedShapeAPTVisitor visitor) {
        visitor.handle(this);
        CallCountResetter resetter = new CallCountResetter();
        this.accept(resetter);
    }
}
