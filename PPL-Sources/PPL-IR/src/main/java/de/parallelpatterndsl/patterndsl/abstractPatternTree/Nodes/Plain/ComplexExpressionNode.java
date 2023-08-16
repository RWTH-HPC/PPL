package de.parallelpatterndsl.patterndsl.abstractPatternTree.Nodes.Plain;

import de.parallelpatterndsl.patterndsl.abstractPatternTree.Visitor.APTVisitor;
import de.parallelpatterndsl.patterndsl.abstractPatternTree.Visitor.CallCountResetter;
import de.parallelpatterndsl.patterndsl.abstractPatternTree.Visitor.ExtendedShapeAPTVisitor;
import de.parallelpatterndsl.patterndsl.expressions.IRLExpression;
import de.parallelpatterndsl.patterndsl.abstractPatternTree.Nodes.PatternNode;
import de.parallelpatterndsl.patterndsl.helperLibrary.DeepCopyHelper;

import java.util.ArrayList;

/**
 * Definition of an expression that contains at least one function call.
 */
public class ComplexExpressionNode extends PatternNode {

    /**
     * The expression it contains.
     */
    private IRLExpression expression;

    public ComplexExpressionNode(IRLExpression expression) {
        this.expression = expression;
    }

    public IRLExpression getExpression() {
        return expression;
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
        long cost = expression.getOperationCount();
        for (PatternNode child: getChildren() ) {
            cost += child.getCost();
        }
        return cost;
    }

    @Override
    public long getLoadStore() {
        long cost = expression.getLoadStores();
        for (PatternNode child: getChildren() ) {
            cost += child.getLoadStore();
        }
        return cost;
    }

    @Override
    public boolean containsSynchronization() {
        return expression.hasExit() || expression.hasProfilingInfo();
    }

    @Override
    public ComplexExpressionNode deepCopy() {

        ComplexExpressionNode result = new ComplexExpressionNode(expression.deepCopy());

        DeepCopyHelper.basicSetup(this, result);

        return result;
    }
}
