package de.parallelpatterndsl.patterndsl.abstractPatternTree.Nodes.Plain;

import de.parallelpatterndsl.patterndsl.abstractPatternTree.Visitor.APTVisitor;
import de.parallelpatterndsl.patterndsl.abstractPatternTree.Visitor.CallCountResetter;
import de.parallelpatterndsl.patterndsl.abstractPatternTree.Visitor.ExtendedShapeAPTVisitor;
import de.parallelpatterndsl.patterndsl.expressions.IRLExpression;
import de.parallelpatterndsl.patterndsl.abstractPatternTree.Nodes.PatternNode;
import de.parallelpatterndsl.patterndsl.helperLibrary.DeepCopyHelper;

import java.util.ArrayList;

/**
 * Class that contains the node for the Sequential-Block Pattern.
 */
public class SimpleExpressionBlockNode extends PatternNode {
    /**
     * Stores an ordered list of simple sequential operations
     * It starts after the last complex pattern and ends before the next complex pattern
     */
    private ArrayList<IRLExpression> expressionList;

    public SimpleExpressionBlockNode(ArrayList<IRLExpression> expressionList) {
        this.expressionList = expressionList;
    }

    public ArrayList<IRLExpression> getExpressionList() {
        return expressionList;
    }

    /**
     * Visitor accept function.
     */
    public void accept(APTVisitor visitor) {
        visitor.handle(this);
    }

    public void accept(ExtendedShapeAPTVisitor visitor) {
        visitor.handle(this);
        CallCountResetter resetter = new CallCountResetter();
        this.accept(resetter);
    }

    @Override
    public long getCost() {
        long cost = 0;
        for (IRLExpression exp: getExpressionList() ) {
            cost += exp.getOperationCount();
        }
        return cost;
    }

    @Override
    public long getLoadStore() {
        long cost = 0;
        for (IRLExpression exp: getExpressionList() ) {
            cost += exp.getLoadStores();
        }
        return cost;
    }

    @Override
    public SimpleExpressionBlockNode deepCopy() {

        ArrayList<IRLExpression> newExpressions = new ArrayList<>();
        for (IRLExpression exp: expressionList ) {
            newExpressions.add(exp.deepCopy());
        }

        SimpleExpressionBlockNode result = new SimpleExpressionBlockNode(newExpressions);

        DeepCopyHelper.basicSetup(this, result);

        return result;
    }
}
