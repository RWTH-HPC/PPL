package de.parallelpatterndsl.patterndsl.MappingTree.Nodes.Plain;

import de.parallelpatterndsl.patterndsl.MappingTree.AbstractMappingTree;
import de.parallelpatterndsl.patterndsl.MappingTree.Nodes.DataControl.DataPlacement;
import de.parallelpatterndsl.patterndsl.MappingTree.Nodes.DataControl.EndPoint;
import de.parallelpatterndsl.patterndsl.MappingTree.Nodes.FunctionMapping;
import de.parallelpatterndsl.patterndsl.MappingTree.Nodes.MappingNode;
import de.parallelpatterndsl.patterndsl.MappingTree.Visitor.AMTVisitor;
import de.parallelpatterndsl.patterndsl.abstractPatternTree.DataElements.ArrayData;
import de.parallelpatterndsl.patterndsl.abstractPatternTree.DataElements.Data;
import de.parallelpatterndsl.patterndsl.abstractPatternTree.DataElements.FunctionInlineData;
import de.parallelpatterndsl.patterndsl.abstractPatternTree.Nodes.Plain.CallNode;
import de.parallelpatterndsl.patterndsl.expressions.OperationExpression;
import de.parallelpatterndsl.patterndsl.expressions.Operator;
import de.parallelpatterndsl.patterndsl.helperLibrary.PredefinedFunctions;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.HashSet;
import java.util.Optional;

/**
 * Defines a function call within the abstract mapping tree.
 */
public class CallMapping extends MappingNode {

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

    public CallMapping(Optional<MappingNode> parent, HashMap<String, Data> variableTable, CallNode aptNode) {
        super(parent, variableTable, aptNode);
        parameterCount = aptNode.getParameterCount();
        functionIdentifier = aptNode.getFunctionIdentifier();
        callExpression = aptNode.getCallExpression();
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

    public int getParameterCount() {
        return parameterCount;
    }

    public String getFunctionIdentifier() {
        return functionIdentifier;
    }

    public FunctionInlineData getCallExpression() {
        return callExpression;
    }

    @Override
    public ArrayList<MappingNode> getChildren() {
        // handle predefined functions
        if (PredefinedFunctions.contains(functionIdentifier)) {
            return new ArrayList<>();
        } else {
            HashMap<String, FunctionMapping> table = AbstractMappingTree.getFunctionTable();
            return table.get(functionIdentifier).getChildren();
        }
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
    public void accept(AMTVisitor visitor) {
        visitor.handle(this);
    }
}
