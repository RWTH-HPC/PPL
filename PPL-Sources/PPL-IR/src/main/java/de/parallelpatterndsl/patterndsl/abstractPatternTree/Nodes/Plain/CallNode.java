package de.parallelpatterndsl.patterndsl.abstractPatternTree.Nodes.Plain;

import de.parallelpatterndsl.patterndsl.abstractPatternTree.AbstractPatternTree;
import de.parallelpatterndsl.patterndsl.abstractPatternTree.DataElements.Data;
import de.parallelpatterndsl.patterndsl.abstractPatternTree.DataElements.DataAccess.DataAccess;
import de.parallelpatterndsl.patterndsl.abstractPatternTree.DataElements.FunctionInlineData;
import de.parallelpatterndsl.patterndsl.abstractPatternTree.Nodes.FunctionNode;
import de.parallelpatterndsl.patterndsl.abstractPatternTree.Nodes.PatternNode;
import de.parallelpatterndsl.patterndsl.abstractPatternTree.Visitor.APTVisitor;
import de.parallelpatterndsl.patterndsl.abstractPatternTree.Visitor.CallCountResetter;
import de.parallelpatterndsl.patterndsl.abstractPatternTree.Visitor.ExtendedShapeAPTVisitor;
import de.parallelpatterndsl.patterndsl.expressions.OperationExpression;
import de.parallelpatterndsl.patterndsl.expressions.Operator;
import de.parallelpatterndsl.patterndsl.helperLibrary.PredefinedFunctions;

import java.util.ArrayList;
import java.util.HashMap;

/**
 * Definition of a function call in the abstract pattern tree.
 */
public class CallNode extends PatternNode{

    /**
     * The number of parameters given.
     */
    private int parameterCount;

    /**
     * The name of the function.
     */
    protected String functionIdentifier;

    /**
     * The expression corresponding to this call.
     */
    private FunctionInlineData callExpression;

    /**
     * A value used by the extended APT Visitor to define the how often this function call was passed up to the current point .
     */
    private int callCount = 0;

    @Override
    public ArrayList<PatternNode> getChildren() {
        // handle predefined functions
        if (PredefinedFunctions.contains(functionIdentifier)) {
            return new ArrayList<>();
        } else {
            HashMap<String, FunctionNode> table = AbstractPatternTree.getFunctionTable();
            return table.get(functionIdentifier).getChildren();
        }
    }

    @Override
    public ArrayList<DataAccess> getInputAccesses() {
        // handle predefined functions
        if (PredefinedFunctions.contains(functionIdentifier)) {
            return new ArrayList<>();
        } else {
            return super.getInputAccesses();
        }
    }

    @Override
    public ArrayList<DataAccess> getOutputAccesses() {
        // handle predefined functions
        if (PredefinedFunctions.contains(functionIdentifier)) {
            return new ArrayList<>();
        } else {
            return super.getOutputAccesses();
        }
    }

    public int getParameterCount() {
        return parameterCount;
    }

    public String getFunctionIdentifier() {
        return functionIdentifier;
    }

    public CallNode(int parameterCount, String functionIdentifier) {
        this.parameterCount = parameterCount;
        this.functionIdentifier = functionIdentifier;
    }

    public FunctionInlineData getCallExpression() {
        return callExpression;
    }

    public void setCallExpression(FunctionInlineData callExpression) {
        this.callExpression = callExpression;
    }

    public int getCallCount() {
        return callCount;
    }

    public void incrementCallCount() {
        callCount++;
    }

    public void resetCallCount() {
        callCount = 0;
    }

    // Not applicable for List Expression.
    public ArrayList<OperationExpression> getArgumentExpressions() {

        ArrayList<OperationExpression> arguments = new ArrayList<>();

        int firstOperand = 1;
        int numOperands = 1;

        int firstOperator = 1;
        int numOperators = 0;

        OperationExpression call = callExpression.getCall();
        for (int i = 0; i < parameterCount; i++) {
            ArrayList<Data> operands = new ArrayList<>();
            ArrayList<Operator> operators = new ArrayList<>();

            boolean nextArgument = false;
            while (!nextArgument) {
                if (call.getOperators().get(firstOperator + numOperators) == Operator.COMMA || call.getOperators().get(firstOperator + numOperators) == Operator.RIGHT_CALL_PARENTHESIS) {
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
