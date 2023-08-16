package de.parallelpatterndsl.patterndsl.abstractPatternTree.Nodes.Functions;

import de.parallelpatterndsl.patterndsl.CombinerFunction;
import de.parallelpatterndsl.patterndsl.abstractPatternTree.DataElements.FunctionInlineData;
import de.parallelpatterndsl.patterndsl.abstractPatternTree.DataElements.FunctionReturnData;
import de.parallelpatterndsl.patterndsl.abstractPatternTree.Nodes.PatternNode;
import de.parallelpatterndsl.patterndsl.abstractPatternTree.Nodes.Plain.ComplexExpressionNode;
import de.parallelpatterndsl.patterndsl.abstractPatternTree.Nodes.Plain.SimpleExpressionBlockNode;
import de.parallelpatterndsl.patterndsl.abstractPatternTree.Visitor.APTVisitor;
import de.parallelpatterndsl.patterndsl.abstractPatternTree.Visitor.CallCountResetter;
import de.parallelpatterndsl.patterndsl.abstractPatternTree.Visitor.ExtendedShapeAPTVisitor;
import de.parallelpatterndsl.patterndsl.expressions.AssignmentExpression;
import de.parallelpatterndsl.patterndsl.expressions.IRLExpression;
import de.parallelpatterndsl.patterndsl.expressions.Operator;
import de.parallelpatterndsl.patterndsl.helperLibrary.DeepCopyHelper;

import java.util.ArrayList;

/**
 * Definition of the reduction pattern node of the abstract pattern tree.
 */
public class ReduceNode extends ParallelNode {


    public ReduceNode(String identifier) {
        super(identifier);
    }

    /**
     * Returns the combiner function of the reduction.
     * @return
     */
    public CombinerFunction getCombinerFunction() {
        PatternNode node = getChildren().get(getChildren().size() - 1);
        if (node instanceof ComplexExpressionNode) {
            IRLExpression exp = ((ComplexExpressionNode) node).getExpression();

            if (exp instanceof AssignmentExpression) {
                if (((AssignmentExpression) exp).getOperator() == Operator.TIMES_ASSIGNMENT) {
                    return CombinerFunction.TIMES;
                } else if (((AssignmentExpression) exp).getOperator() == Operator.PLUS_ASSIGNMENT) {
                    return CombinerFunction.PLUS;
                } else if (((AssignmentExpression) exp).getOperator() == Operator.ASSIGNMENT) {
                    if (((AssignmentExpression) exp).getRhsExpression().getOperands().get(0) instanceof FunctionInlineData) {
                        FunctionInlineData function = (FunctionInlineData) ((AssignmentExpression) exp).getRhsExpression().getOperands().get(0);
                        FunctionReturnData returnData = (FunctionReturnData) function.getCall().getOperands().get(0);
                        if (returnData.getIdentifier().equals("max")) {
                            return CombinerFunction.MAX;
                        } else if (returnData.getIdentifier().equals("min")) {
                            return CombinerFunction.MIN;
                        }
                    }
                }
            }
        } else if (node instanceof SimpleExpressionBlockNode) {
            IRLExpression exp = ((SimpleExpressionBlockNode) node).getExpressionList().get(((SimpleExpressionBlockNode) node).getExpressionList().size() - 1);
            if (exp instanceof AssignmentExpression) {
                if (((AssignmentExpression) exp).getOperator() == Operator.TIMES_ASSIGNMENT) {
                    return CombinerFunction.TIMES;
                } else if (((AssignmentExpression) exp).getOperator() == Operator.PLUS_ASSIGNMENT) {
                    return CombinerFunction.PLUS;
                }
            }
        }
        return CombinerFunction.FAILURE;
    }

    @Override
    public ReduceNode deepCopy() {
        ReduceNode result = new ReduceNode(getIdentifier());

        DeepCopyHelper.basicPatternSetup(this, result);

        return result;
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
