package de.parallelpatterndsl.patterndsl.abstractPatternTree.Nodes.Plain;

import de.parallelpatterndsl.patterndsl.abstractPatternTree.AbstractPatternTree;
import de.parallelpatterndsl.patterndsl.abstractPatternTree.DataElements.Data;
import de.parallelpatterndsl.patterndsl.abstractPatternTree.DataElements.FunctionInlineData;
import de.parallelpatterndsl.patterndsl.abstractPatternTree.DataElements.PrimitiveDataTypes;
import de.parallelpatterndsl.patterndsl.abstractPatternTree.Nodes.FunctionNode;
import de.parallelpatterndsl.patterndsl.abstractPatternTree.Nodes.Functions.RecursionNode;
import de.parallelpatterndsl.patterndsl.abstractPatternTree.Nodes.PatternNode;
import de.parallelpatterndsl.patterndsl.abstractPatternTree.Nodes.Plain.AdditionalArguments.AdditionalArguments;
import de.parallelpatterndsl.patterndsl.abstractPatternTree.Visitor.APTVisitor;
import de.parallelpatterndsl.patterndsl.abstractPatternTree.Visitor.CallCountResetter;
import de.parallelpatterndsl.patterndsl.abstractPatternTree.Visitor.ExtendedShapeAPTVisitor;
import de.parallelpatterndsl.patterndsl.expressions.AssignmentExpression;
import de.parallelpatterndsl.patterndsl.expressions.IRLExpression;
import de.parallelpatterndsl.patterndsl.expressions.OperationExpression;
import de.parallelpatterndsl.patterndsl.expressions.Operator;
import de.se_rwth.commons.logging.Log;

import java.util.ArrayList;
import java.util.HashMap;

/**
 * Definition of a parallel pattern call node of the abstract pattern tree.
 * The first n element are accounted to the additional arguments, where n is defined by additionalArgumentCount.
 * The following element describes the assignment of the call.
 */
public class ParallelCallNode extends CallNode {

    /**
     * The number of meta arguments within the parallel call notation.
     * The number denotes how many elements are accounted to the additional arguments within the child nodes.
     */
    private int additionalArgumentCount;

    /**
     * Defines a set of meta information necessary in the optimization and the generation.
     */
    private ArrayList<AdditionalArguments> additionalArguments;

    public ParallelCallNode(int parameterCount, String functionIdentifier, int additionalArgumentCount) {
        super(parameterCount, functionIdentifier);
        this.additionalArgumentCount = additionalArgumentCount;

        setCallExpression(new FunctionInlineData("", PrimitiveDataTypes.VOID, new OperationExpression(new ArrayList<>(),new ArrayList<>()), -1));
    }

    public int getAdditionalArgumentCount() {
        return additionalArgumentCount;
    }


    @Override
    public ArrayList<PatternNode> getChildren() {
        HashMap<String, FunctionNode> table = AbstractPatternTree.getFunctionTable();
        ArrayList<PatternNode> allChildren = new ArrayList<>(children);

        FunctionNode function = table.get(functionIdentifier);

        if (!(function instanceof RecursionNode)) {
            allChildren.addAll(function.getChildren());
        }

        return allChildren;
    }

    public ArrayList<AdditionalArguments> getAdditionalArguments() {
        return additionalArguments;
    }

    public void setAdditionalArguments(ArrayList<AdditionalArguments> additionalArguments) {
        this.additionalArguments = additionalArguments;
    }

    @Override
    public ArrayList<OperationExpression> getArgumentExpressions() {

        ArrayList<OperationExpression> arguments = new ArrayList<>();

        int firstOperand = 1;
        int numOperands = 1;

        int firstOperator = 1;
        int numOperators = 0;

        OperationExpression call;

        if (children.get(0) instanceof ComplexExpressionNode) {
            IRLExpression exp =  ((ComplexExpressionNode) children.get(0)).getExpression();
            if (exp instanceof AssignmentExpression) {
                call = ((AssignmentExpression) exp).getRhsExpression();
            } else {
                Log.error("Parallel Call expression not correctly generated!  " + super.getFunctionIdentifier());
                throw new RuntimeException("Critical error!");
            }
        } else {
            Log.error("Parallel Call expression not correctly generated!  " + super.getFunctionIdentifier());
            throw new RuntimeException("Critical error!");
        }

        for (int i = 0; i < super.getParameterCount(); i++) {
            ArrayList<Data> operands = new ArrayList<>();
            ArrayList<Operator> operators = new ArrayList<>();

            boolean nextArgument = false;
            while (!nextArgument) {
                if (call.getOperators().get(firstOperator + numOperators) == Operator.COMMA || call.getOperators().get(firstOperator + numOperators) == Operator.RIGHT_CALL_PARENTHESIS ) {
                    nextArgument = true;
                } else { //if(Operator.arity(call.getOperators().get(firstOperator + numOperators)) == 2) {
                    numOperators++;
                }

            }
            if (numOperators != 0) {
                for (int j = firstOperator; j < firstOperator + numOperators; j++) {
                    operators.add(call.getOperators().get(j));
                }
            }

            operators.remove(Operator.COMMA);

            for (Operator operator: operators ) {
                if (Operator.arity(operator) == 2) {
                    numOperands++;
                }
            }

            for (int j = firstOperand; j < firstOperand + numOperands; j++) {
                operands.add(call.getOperands().get(j));
            }

            firstOperand += numOperands;
            firstOperator += numOperators + 1;

            numOperands = 1;
            numOperators = 0;

            arguments.add(new OperationExpression(operands, operators));

        }
        return arguments;
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
