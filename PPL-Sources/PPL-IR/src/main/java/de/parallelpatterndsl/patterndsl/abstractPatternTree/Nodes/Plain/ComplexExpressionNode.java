package de.parallelpatterndsl.patterndsl.abstractPatternTree.Nodes.Plain;

import de.parallelpatterndsl.patterndsl.abstractPatternTree.Visitor.APTVisitor;
import de.parallelpatterndsl.patterndsl.abstractPatternTree.Visitor.CallCountResetter;
import de.parallelpatterndsl.patterndsl.abstractPatternTree.Visitor.ExtendedShapeAPTVisitor;
import de.parallelpatterndsl.patterndsl.expressions.IRLExpression;
import de.parallelpatterndsl.patterndsl.abstractPatternTree.Nodes.PatternNode;

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
}
