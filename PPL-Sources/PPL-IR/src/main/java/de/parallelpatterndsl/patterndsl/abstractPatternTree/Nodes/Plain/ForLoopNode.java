package de.parallelpatterndsl.patterndsl.abstractPatternTree.Nodes.Plain;

import de.parallelpatterndsl.patterndsl.abstractPatternTree.DataElements.Data;
import de.parallelpatterndsl.patterndsl.abstractPatternTree.DataElements.LiteralData;
import de.parallelpatterndsl.patterndsl.abstractPatternTree.Visitor.APTVisitor;
import de.parallelpatterndsl.patterndsl.abstractPatternTree.Visitor.CallCountResetter;
import de.parallelpatterndsl.patterndsl.abstractPatternTree.Visitor.ExtendedShapeAPTVisitor;
import de.parallelpatterndsl.patterndsl.expressions.AssignmentExpression;
import de.parallelpatterndsl.patterndsl.expressions.OperationExpression;
import de.parallelpatterndsl.patterndsl.expressions.Operator;

/**
 * Definition of a for-loop of the abstract pattern tree.
 * The first three elements in the children array are the assignment of the control variable, the condition and the update expression.
 */
public class ForLoopNode extends LoopNode {

    private int numIterations = -1;

    /**
     * The Variable used for the Loop iteration.
     */
    private Data loopControlVariable;

    public ForLoopNode(Data loopControlVariable) {
        this.loopControlVariable = loopControlVariable;
    }

    public Data getLoopControlVariable() {
        return loopControlVariable;
    }

    /**
     * generates the number of iterations.
     * @return
     */
    private int genNumIterations() {
        int start = 0;
        int end = 0;
        int total = 1;

        // Test and get the starting value for the loop iteration.
        if (children.get(0) instanceof ComplexExpressionNode) {

            if (((ComplexExpressionNode) children.get(0)).getExpression() instanceof AssignmentExpression) {
                AssignmentExpression exp = (AssignmentExpression) ((ComplexExpressionNode) children.get(0)).getExpression();
                if (exp.getRhsExpression().getOperators().size() == 0 && exp.getRhsExpression().getOperands().size() == 1) {
                    if (exp.getRhsExpression().getOperands().get(0) instanceof LiteralData) {
                        if (((LiteralData) exp.getRhsExpression().getOperands().get(0)).getValue() instanceof Integer) {
                            start = (int) ((LiteralData) exp.getRhsExpression().getOperands().get(0)).getValue();
                        }
                    }
                }
            }
        }

        // Test and get the end value for the Loop iteration
        if (children.get(1) instanceof ComplexExpressionNode) {

            if (((ComplexExpressionNode) children.get(1)).getExpression() instanceof OperationExpression) {
                OperationExpression exp = (OperationExpression) ((ComplexExpressionNode) children.get(1)).getExpression();
                if (exp.getOperators().size() == 1 && exp.getOperands().size() == 2) {
                    if (exp.getOperands().get(1) instanceof LiteralData) {
                        if (((LiteralData) exp.getOperands().get(1)).getValue() instanceof Integer) {
                            end = (int) ((LiteralData) exp.getOperands().get(1)).getValue();
                            if (exp.getOperators().get(0) == Operator.LESS) {
                                end--;
                            } else if (exp.getOperators().get(0) == Operator.GREATER) {
                                end++;
                            }
                        }
                    }
                }
            }
        }

        // get the total value depending on the update rule
        if (children.get(2) instanceof ComplexExpressionNode) {
            if (((ComplexExpressionNode) children.get(2)).getExpression() instanceof AssignmentExpression) {
                AssignmentExpression exp = (AssignmentExpression) ((ComplexExpressionNode) children.get(2)).getExpression();
                if (exp.getOperator() == Operator.INCREMENT) {
                    total = end - start + 1;
                } else if (exp.getOperator() == Operator.DECREMENT) {
                    total = start - end + 1;
                }
            }
        }

        if (total < 0) {
            total *= -1;
        }

        return total;
    }


    @Override
    public int getNumIterations() {
        if (numIterations == -1) {
            numIterations = genNumIterations();
        }
        return numIterations;
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
