package de.parallelpatterndsl.patterndsl.printer;

import de.parallelpatterndsl.patterndsl.MappingTree.AbstractMappingTree;
import de.parallelpatterndsl.patterndsl.MappingTree.Nodes.Function.SerialMapping;
import de.parallelpatterndsl.patterndsl.MappingTree.Nodes.FunctionMapping;
import de.parallelpatterndsl.patterndsl.MappingTree.Nodes.ParallelCalls.ParallelCallMapping;
import de.parallelpatterndsl.patterndsl.MappingTree.Nodes.Plain.CallMapping;
import de.parallelpatterndsl.patterndsl.abstractPatternTree.AbstractPatternTree;
import de.parallelpatterndsl.patterndsl.abstractPatternTree.DataElements.*;
import de.parallelpatterndsl.patterndsl.abstractPatternTree.Nodes.FunctionNode;
import de.parallelpatterndsl.patterndsl.expressions.AssignmentExpression;
import de.parallelpatterndsl.patterndsl.expressions.IRLExpression;
import de.parallelpatterndsl.patterndsl.expressions.OperationExpression;
import de.parallelpatterndsl.patterndsl.expressions.Operator;
import de.parallelpatterndsl.patterndsl.helperLibrary.PredefinedFunctions;
import de.se_rwth.commons.logging.Log;

import java.util.ArrayList;
import java.util.HashMap;

/**
 * This class implements the conversion of an IRL expression to a C++ expression, while also accounting for potential inlining of function calls.
 */

public class CppExpressionPrinter {

    private static boolean inlineFunctions = false;

    private HashMap<Operator, String> operator2Cpp;

    private static CppExpressionPrinter instance;

    private static HashMap<Data, OperationExpression> parallelVariableInlining = new HashMap<>();

    private static ArrayList<CallMapping> activeCalls;

    private static ArrayList<ParallelCallMapping> activePatterns;

    private static boolean needsMPI;

    private static AbstractMappingTree AMT;

    private boolean withinInlineGeneration;

    private ArrayData currentAssignment;

    public CppExpressionPrinter() {
        withinInlineGeneration = false;
        operator2Cpp = new HashMap<>();
        operator2Cpp.put(Operator.PLUS, " + ");
        operator2Cpp.put(Operator.MINUS, " - ");
        operator2Cpp.put(Operator.MULTIPLICATION, " * ");
        operator2Cpp.put(Operator.DIVIDE, " / ");
        operator2Cpp.put(Operator.MODULO, " % ");
        operator2Cpp.put(Operator.INCREMENT, "++");
        operator2Cpp.put(Operator.DECREMENT, "--");
        operator2Cpp.put(Operator.LESS, " < ");
        operator2Cpp.put(Operator.LESS_OR_EQUAL, " <= ");
        operator2Cpp.put(Operator.EQUAL, " == ");
        operator2Cpp.put(Operator.NOT_EQUAL, " != ");
        operator2Cpp.put(Operator.GREATER, " > ");
        operator2Cpp.put(Operator.GREATER_OR_EQUAL, " >= ");
        operator2Cpp.put(Operator.LOGICAL_NOT, "!");
        operator2Cpp.put(Operator.LOGICAL_AND, " && ");
        operator2Cpp.put(Operator.LOGICAL_OR, " || ");
        operator2Cpp.put(Operator.BITWISE_NOT, "~");
        operator2Cpp.put(Operator.BITWISE_OR, " | ");
        operator2Cpp.put(Operator.BITWISE_AND, " & ");
        operator2Cpp.put(Operator.LEFT_PARENTHESIS, "(");
        operator2Cpp.put(Operator.RIGHT_PARENTHESIS, ")");
        operator2Cpp.put(Operator.ASSIGNMENT, " = ");
        operator2Cpp.put(Operator.PLUS_ASSIGNMENT, " += ");
        operator2Cpp.put(Operator.MINUS_ASSIGNMENT, " -= ");
        operator2Cpp.put(Operator.TIMES_ASSIGNMENT, " *= ");
        operator2Cpp.put(Operator.BIT_AND_ASSIGNMENT, " &= ");
        operator2Cpp.put(Operator.BIT_OR_ASSIGNMENT, " |= ");
        operator2Cpp.put(Operator.LEFT_ARRAY_ACCESS, "");
        operator2Cpp.put(Operator.RIGHT_ARRAY_ACCESS, "");
        operator2Cpp.put(Operator.LEFT_CALL_PARENTHESIS, "(");
        operator2Cpp.put(Operator.RIGHT_CALL_PARENTHESIS, ")");
        operator2Cpp.put(Operator.COMMA, ", ");
        operator2Cpp.put(Operator.LEFT_ARRAY_DEFINITION, "");
        operator2Cpp.put(Operator.RIGHT_ARRAY_DEFINITION, "");
    }

    public static void setAMT(AbstractMappingTree AMT) {
        CppExpressionPrinter.AMT = AMT;
    }

    public static String doPrintExpression(IRLExpression expression, boolean inlineFunctionsCurrent, ArrayList<CallMapping> activeCallList, ArrayList<ParallelCallMapping> activePatternList) {
        inlineFunctions = inlineFunctionsCurrent;
        activeCalls = activeCallList;
        activePatterns = activePatternList;
        if (instance == null) {
            instance = new CppExpressionPrinter();
        }

        return instance.printExpression(expression);
    }

    private String printExpression(IRLExpression exp) {
        if (exp instanceof AssignmentExpression) {
            return printAssignmentExpression((AssignmentExpression) exp);
        } else if (exp instanceof OperationExpression) {
            return printOperationExpression((OperationExpression) exp);
        }

        Log.error("Unknown type of expression!");
        throw new RuntimeException("Critical error!");
    }

    /**
     * Tests if the function named identifier takes arrays as its arguments.
     * String identifier
     *
     * @return
     */
    private boolean hasListArguments(String identifier) {
        if (PredefinedFunctions.contains(identifier)) {
            return false;
        }
        FunctionMapping function = AbstractMappingTree.getFunctionTable().get(identifier);
        if (function instanceof SerialMapping) {
            if (((SerialMapping) function).isList()) {
                return true;
            }
        }
        return function.getArgumentValues().stream().anyMatch(c -> c instanceof ArrayData);
    }

    public static void setNeedsMPI(boolean needsMPI) {
        CppExpressionPrinter.needsMPI = needsMPI;
    }

    /**
     * Prints the function call encapsulated in a function inline data element.
     * If the call will be inlined, the call will be replaced by a fresh variable.
     *
     * @param data
     * @return
     */
    private String printFunctionInlineData(FunctionInlineData data) {
        StringBuilder builder = new StringBuilder();

        if (data.getIdentifier().equals("Get_Size_Of_Vector")) {
            if (data.getCall().getOperands().get(1) instanceof ArrayData) {
                ArrayData arrayData = (ArrayData) data.getCall().getOperands().get(1);
                builder.append(arrayData.getShape().get((int) data.getCall().getOperators().stream().filter(c -> c == Operator.LEFT_ARRAY_ACCESS).count()));
                return builder.toString();
            }
        }

        if (data.getIdentifier().equals("Element_Exists_In_Vector")) {
            builder.append(printOperationExpression(data.getCall()));
            builder.deleteCharAt(builder.length() - 1);
            builder.append(", ");
            builder.append(1);
            for (int dimension : data.getCall().getShape()) {
                builder.append("LL * ");
                builder.append(dimension);
            }
            builder.append("LL)");
            return builder.toString();
        }

        if (data.getCall().getOperands().get(0).getIdentifier().equals("init_List")) {

            OperationExpression expression = data.getCall();
            int length = expression.getOperands().size();
            if (expression.getOperators().get(expression.getOperators().size() - 2) == Operator.COMMA) {
                length = expression.getOperands().size() - 1;
            }

            builder.append("Init_List(");

            if (data.getCall().getOperators().get(data.getCall().getOperators().size() - 2) == Operator.COMMA) {
                addPrintData(builder, data.getCall().getOperands().get(data.getCall().getOperands().size() - 1));
                builder.append(", ");
            }

            addPrintData(builder, currentAssignment);
            builder.append(", ");

            for (int i = 1; i < length; i++) {
                builder.append(((LiteralData) expression.getOperands().get(i)).getValue());
                builder.append("LL * ");
            }
            builder.append("1LL");
            builder.append(")");


            /*
            builder.append("(");
            builder.append(CppTypesPrinter.doPrintType(currentAssignment.getTypeName()));
            builder.append("*) std::malloc(");
            for (int i = 1; i < length; i++) {
                builder.append(((LiteralData) expression.getOperands().get(i)).getValue());
                builder.append(" * ");
            }
            builder.append("sizeof(");
            builder.append(CppTypesPrinter.doPrintType(currentAssignment.getTypeName()));
            builder.append("))");*/
        } else if ((inlineFunctions || hasListArguments(data.getCall().getOperands().get(0).getIdentifier())) && !PredefinedFunctions.contains(data.getCall().getOperands().get(0).getIdentifier())) {

            boolean printIdentifier = true;
            if (data.getCall().getOperands().get(0) instanceof FunctionReturnData) {
                if (!AbstractMappingTree.getFunctionTable().containsKey(data.getCall().getOperands().get(0).getIdentifier())) {
                    int a = 0;
                }
                if (((SerialMapping) AbstractMappingTree.getFunctionTable().get(data.getCall().getOperands().get(0).getIdentifier())).getReturnType() == PrimitiveDataTypes.VOID) {
                    printIdentifier = false;
                }
            }

            if (printIdentifier) {
                builder.append(data.getIdentifier());
                builder.append("_");
                builder.append(data.getInlineEnding());
            }

        } else {
            builder.append(printOperationExpression(data.getCall()));
        }
        return builder.toString();
    }

    /**
     * Prints an Assignment Expression.
     *
     * @param exp
     * @return
     */
    private String printAssignmentExpression(AssignmentExpression exp) {
        StringBuilder builder = new StringBuilder();
        boolean hasSubArrayChange = false;
        boolean isInitializer = false;
        boolean hasInitValue = false;

        // test if the initializer is used
        if (!exp.getRhsExpression().getOperands().isEmpty()) {
            if (exp.getRhsExpression().getOperands().get(0) instanceof FunctionInlineData) {
                FunctionInlineData assignment = (FunctionInlineData) exp.getRhsExpression().getOperands().get(0);
                if (assignment.getCall().getOperands().get(0).getIdentifier().startsWith("init_List")) {
                    isInitializer = true;
                    if (assignment.getCall().getOperators().get(assignment.getCall().getOperators().size() - 2) == Operator.COMMA) {
                        hasInitValue = true;
                    }
                }
            }
        }
        if (exp.getOutputElement().isInlinedParameter()) {
            isInitializer = true;
        }

        // Create flattened access scheme.

        if (exp.getOutputElement() instanceof ArrayData) {
            currentAssignment = (ArrayData) exp.getOutputElement();
            if (((ArrayData) exp.getOutputElement()).getShape().size() > exp.getAccessScheme().size()) {
                hasSubArrayChange = true;
                if (exp.getRhsExpression().getOperators().size() > 0 && exp.getAccessScheme().size() == 0) {
                    if (exp.getRhsExpression().getOperators().get(0) == Operator.LEFT_ARRAY_DEFINITION) {
                        builder.append(CppTypesPrinter.doPrintType(exp.getOutputElement().getTypeName()));
                        builder.append(" ");
                        addPrintData(builder, exp.getOutputElement());
                        builder.append("[]");
                        builder.append(operator2Cpp.get(exp.getOperator()));
                        builder.append("{");
                        for (Data data : exp.getRhsExpression().getOperands()) {
                            if (data instanceof LiteralData) {
                                builder.append(((LiteralData) data).getValue());
                                builder.append(", ");
                            } else {
                                Log.error("Expected Literal in array definition!");
                                throw new RuntimeException("Critical error!");
                            }
                        }
                        builder.deleteCharAt(builder.length() - 2);
                        builder.append("}");
                        return builder.toString();
                    }
                }

                if (!exp.getShape().isEmpty() && !isInitializer) {
                    builder.append("Set_Partial_Array");
                    if (exp.getOutputElement().getIdentifier().startsWith("inlineReturn_")) {
                        builder.append("_Ref");
                        exp.getRhsExpression().getOperands().stream().filter(x -> x instanceof ArrayData).forEach(x -> x.setInlinedReturnValue(true));
                    }
                    builder.append("(");
                    if (!exp.getAccessScheme().isEmpty() || exp.getOutputElement().getIdentifier().startsWith("inlineReturn_")) {
                        builder.append("&");
                    }
                    addPrintAssignment(builder, exp);
                    builder.append(", ");
                    boolean hasRead = false;
                    if (exp.getRhsExpression().getOperands().size() > 1) {
                        if (!exp.isHasIOData()) {
                            builder.append("&");
                        } else {
                            //Choose which data type is read
                            if (exp.getOutputElement().getTypeName() == PrimitiveDataTypes.INTEGER_32BIT) {
                                hasRead = true;
                                builder.append("&i_");
                            } else if (exp.getOutputElement().getTypeName() == PrimitiveDataTypes.DOUBLE) {
                                hasRead = true;
                                builder.append("&d_");
                            } else if (exp.getOutputElement().getTypeName() == PrimitiveDataTypes.FLOAT) {
                                hasRead = true;
                                builder.append("&f_");
                            } else {
                                hasRead = true;
                                builder.append("&");
                            }
                        }
                    }
                    builder.append(printOperationExpression(exp.getRhsExpression()));
                    if (hasRead) {
                        if (exp.getRhsExpression().getOperands().size() > 1) {
                            builder.deleteCharAt(builder.length() - 1);
                            builder.append(", 1");
                            for (int i = exp.getAccessScheme().size(); i < ((ArrayData) exp.getOutputElement()).getShape().size(); i++) {
                                builder.append("LL * ");
                                long targetShape = ((ArrayData) exp.getOutputElement()).getShape().get(i);
                                builder.append(targetShape);
                            }
                            builder.append("LL)");
                        }
                        builder.append("[0]");
                    }
                    builder.append(", ");
                    builder.append(1);
                    for (int i = exp.getAccessScheme().size(); i < ((ArrayData) exp.getOutputElement()).getShape().size(); i++) {
                        builder.append("LL * ");
                        long targetShape = ((ArrayData) exp.getOutputElement()).getShape().get(i);
                        builder.append(targetShape);
                    }
                    builder.append("LL)");
                } else {
                    addPrintData(builder, exp.getOutputElement());
                }
            } else {
                addPrintAssignment(builder, exp);
            }
        } else {
            //Part for primitive data elements
            if (parallelVariableInlining.containsKey(exp.getOutputElement())) {
                OperationExpression operationExpression = parallelVariableInlining.get(exp.getOutputElement());
                withinInlineGeneration = true;
                builder.append(printOperationExpression(operationExpression));
                withinInlineGeneration = false;
            } else {
                addPrintData(builder, exp.getOutputElement());
            }
        }


        if (!hasSubArrayChange || exp.getShape().isEmpty() || isInitializer) {
            builder.append(operator2Cpp.get(exp.getOperator()));
            builder.append(printOperationExpression(exp.getRhsExpression()));
        }
        if (hasInitValue) {

        }

        return builder.toString();
    }

    /**
     * Prints an OperationExpression.
     *
     * @param exp
     * @return
     */
    private String printOperationExpression(OperationExpression exp) {
        StringBuilder builder = new StringBuilder();

        boolean doArrayWrite = false;

        int currentOperator = 0;
        boolean finished = false;
        boolean isExitCall = false;

        int i = 0;
        while (i < exp.getOperands().size()) {
            int nextOperator = exp.getRightOperandContextIndex(i);
            if (nextOperator == -1) {
                for (int j = currentOperator; j < exp.getOperators().size(); j++) {
                    builder.append(operator2Cpp.get(exp.getOperators().get(j)));
                }
                finished = true;
            } else if (nextOperator > currentOperator) {
                for (int j = currentOperator; j < nextOperator; j++) {
                    if (!isExitCall || exp.getOperators().get(j) != Operator.LEFT_CALL_PARENTHESIS) {
                        builder.append(operator2Cpp.get(exp.getOperators().get(j)));
                        isExitCall = false;
                    }
                }
                currentOperator = nextOperator;
            }
            Data operand = exp.getOperands().get(i);

            // true if the index access expression was already started by an inlined variable.
            boolean isOpened = false;
            // stores the current starting point for an index access expression. This is used to generate the address to a sub array.
            int start = builder.length();
            if (parallelVariableInlining.containsKey(operand)) {
                OperationExpression operationExpression = parallelVariableInlining.get(exp.getOperands().get(i));
                withinInlineGeneration = true;
                builder.append(printOperationExpression(operationExpression));
                withinInlineGeneration = false;

                if (operand instanceof ArrayData && exp.getOperators().size() > 0) {
                    if (exp.getOperators().get(currentOperator) == Operator.LEFT_ARRAY_ACCESS) {
                        isOpened = true;
                        if (builder.substring(builder.length() - 1).equals("]")) {
                            builder.deleteCharAt(builder.length() - 1);
                            builder.append(" + ");
                        } else {
                            builder.append("[");
                        }
                    }
                }
            } else {
                // write content of an array to file if only an array is defined.
                if (operand == null) {
                    int a = 0;
                    //TODO: look into inlining (functions) for missing variables.
                }
                if (operand.getIdentifier().equals("write") && exp.getOperands().size() == 3) {
                    if (exp.getOperands().get(2) instanceof ArrayData) {
                        builder.append("array_");
                        doArrayWrite = true;
                    }
                }
                addPrintData(builder, operand);
                if (operand.getIdentifier().equals("exit")) {
                    isExitCall = true;
                }

            }
            if (currentOperator != exp.getOperators().size()) {
                if (exp.getOperators().get(currentOperator) == Operator.LEFT_ARRAY_ACCESS && operand instanceof ArrayData) {
                    if (!isOpened) {
                        builder.append("[");
                    }

                    // Array access recombination.
                    for (int j = 0; j < ((ArrayData) operand).getShape().size(); j++) {
                        for (int k = j + 1; k < ((ArrayData) operand).getShape().size(); k++) {
                            builder.append(((ArrayData) operand).getShape().get(k));
                            builder.append("LL * ");
                        }
                        builder.append("(");

                        if (currentOperator > 0) {
                            if (exp.getOperators().get(currentOperator - 1) != Operator.RIGHT_PARENTHESIS && exp.getOperators().get(currentOperator - 1) != Operator.RIGHT_ARRAY_ACCESS) {
                                i++;
                            }
                        } else {
                            i++;
                        }

                        boolean isOpeningAccess = true;
                        while (exp.getOperators().get(currentOperator) != Operator.RIGHT_ARRAY_ACCESS) {
                            if (exp.getOperators().get(currentOperator) == Operator.LEFT_ARRAY_ACCESS && isOpeningAccess) {
                                currentOperator++;
                                isOpeningAccess = false;
                                if (exp.getOperators().get(currentOperator) == Operator.RIGHT_ARRAY_ACCESS) {
                                    if (i < exp.getOperands().size()) {
                                        addPrintData(builder, exp.getOperands().get(i));
                                        i++;
                                    }
                                }
                                continue;
                            }

                            // Cover the different possible cases in an array access
                            if (exp.getOperators().get(currentOperator) == Operator.RIGHT_PARENTHESIS && exp.getOperators().get(currentOperator - 1) != Operator.RIGHT_PARENTHESIS) {
                                addPrintData(builder, exp.getOperands().get(i));
                                builder.append(operator2Cpp.get(exp.getOperators().get(currentOperator)));
                                currentOperator++;
                                i++;
                            } else if (Operator.arity(exp.getOperators().get(currentOperator)) == 1) {
                                builder.append(operator2Cpp.get(exp.getOperators().get(currentOperator)));
                                currentOperator++;
                            } else if (Operator.arity(exp.getOperators().get(currentOperator)) == 2) {
                                if (exp.getOperators().get(currentOperator - 1) != Operator.RIGHT_PARENTHESIS) {
                                    addPrintData(builder, exp.getOperands().get(i));
                                    i++;
                                }
                                builder.append(operator2Cpp.get(exp.getOperators().get(currentOperator)));
                                currentOperator++;

                            }

                            if (exp.getOperators().get(currentOperator) == Operator.RIGHT_ARRAY_ACCESS) {
                                if (exp.getOperators().get(currentOperator - 1) != Operator.RIGHT_PARENTHESIS) {
                                    addPrintData(builder, exp.getOperands().get(i));
                                    i++;
                                }
                            }
                        }

                        builder.append(")");

                        // continue with the next continuous access or break if there is none.
                        if (currentOperator < exp.getOperators().size() - 1) {
                            if (exp.getOperators().get(currentOperator + 1) == Operator.LEFT_ARRAY_ACCESS) {
                                builder.append(" + ");
                                currentOperator++;
                            } else {
                                //Not necessary anymore TODO: wait until no bugs are found
                                /*if (j + 1 != ((ArrayData) operand).getShape().size() && !withinInlineGeneration) {
                                    //builder.insert(start, " &");
                                }*/
                                break;
                            }
                        } else {
                            // Not necessary anymore TODO: wait until no bugs are found
                            /*if (j + 1 != ((ArrayData) operand).getShape().size() && !withinInlineGeneration) {
                                builder.insert(start, " &");
                            }*/
                            break;
                        }
                    }
                    builder.append("]");
                } else {
                    i++;
                }
            } else {
                i++;
            }
        }
        if (!finished) {
            for (int j = currentOperator; j < exp.getOperators().size(); j++) {
                builder.append(operator2Cpp.get(exp.getOperators().get(j)));
            }
        }

        if (doArrayWrite) {
            builder.delete(builder.length() - 1, builder.length());
            builder.append(", 1LL");
            for (Integer dim : ((ArrayData) exp.getOperands().get(2)).getShape() ) {
                builder.append(" * ");
                builder.append(dim);
            }
            builder.append(")");
        }

        return builder.toString();
    }

    /**
     * adds the data element to the printed string.
     *
     * @param builder
     * @param data
     */
    private StringBuilder addPrintData(StringBuilder builder, Data data) {
        if (data instanceof FunctionInlineData) {
            builder.append(printFunctionInlineData((FunctionInlineData) data));
        } else if (data instanceof LiteralData) {
            if (((LiteralData) data).getValue() instanceof Character) {
                builder.append("'");
            }
            builder.append(((LiteralData) data).getValue());
            if (((LiteralData) data).getValue() instanceof Character) {
                builder.append("'");
            }
            if (data.getTypeName() == PrimitiveDataTypes.INTEGER_64BIT) {
                builder.append("LL");
            }
            if (data.getTypeName() == PrimitiveDataTypes.FLOAT) {
                builder.append("f");
            }
        } else if (data instanceof FunctionReturnData) {
            if (data.getIdentifier().equals("exit")) {
                if (needsMPI) {
                    builder.append("MPI_Abort(MPI_COMM_WORLD, ");
                } else {
                    builder.append("exit(");
                }
            } else if (data.getIdentifier().equals("rand")) {
                if (!activePatterns.isEmpty()) {
                    if (activePatterns.get(0).getExecutor().getParent().getType().equals("GPU")) {
                        builder.append("randomizerGPU");
                    } else {
                        builder.append("randomizer");
                    }
                } else {
                    builder.append("randomizer");
                }
            } else {
                builder.append(data.getIdentifier());
                if (inlineFunctions) {
                    builder.append("");
                }
            }
        } else {
            // Handle global vars
            if (AMT.getGlobalVariableTable().containsKey(data.getIdentifier())) {
                /*for (IRLExpression assign: AMT.getGlobalAssignments()) {
                    if (assign instanceof AssignmentExpression) {
                        if (((AssignmentExpression) assign).getOutputElement() == data) {
                            builder.append(printOperationExpression(((AssignmentExpression) assign).getRhsExpression()));
                            return builder;
                        }
                    }
                }*/
            }

            builder.append(data.getIdentifier());

            if (!activeCalls.isEmpty()) {
                FunctionNode function = AbstractPatternTree.getFunctionTable().get(activeCalls.get(activeCalls.size() - 1).getFunctionIdentifier());
                boolean doInline = CppPlainNodePrinter.calculateInlining(activeCalls.get(activeCalls.size() - 1));


                if ((doInline || inlineFunctions) && !AbstractPatternTree.getInstance().getGlobalVariableTable().containsKey(data.getIdentifier()) && !data.getIdentifier().startsWith("INDEX") && !(data.getIdentifier().endsWith("[0]"))) {
                    if(!data.hasInlineIdentifier() && !(data instanceof IOData)) {
                        builder.append("_");
                        builder.append(activeCalls.get(activeCalls.size() - 1).getCallExpression().getInlineEnding());
                    }
                }

            }
        }
        return builder;
    }

    /**
     * adds the left hand side of an assignment expression to the builder.
     *
     * @param builder
     * @param exp
     * @return
     */
    private StringBuilder addPrintAssignment(StringBuilder builder, AssignmentExpression exp) {
        // Test if the data element is an inlined parallel array
        if (parallelVariableInlining.containsKey(exp.getOutputElement())) {
            OperationExpression operationExpression = parallelVariableInlining.get(exp.getOutputElement());
            withinInlineGeneration = true;
            builder.append(printOperationExpression(operationExpression));
            withinInlineGeneration = false;
            if (!operationExpression.getOperators().isEmpty()) {
                if (operationExpression.getOperators().get(operationExpression.getOperators().size() - 1) == Operator.RIGHT_ARRAY_ACCESS) {
                    builder.deleteCharAt(builder.length() - 1);
                    builder.append(" + ");
                }
            } else {
                if (!exp.getAccessScheme().isEmpty()) {
                    builder.append("[");
                }
            }

        } else {
            addPrintData(builder, exp.getOutputElement());
            if (!exp.getAccessScheme().isEmpty()) {
                builder.append("[");
            }
        }
        for (int i = 0; i < exp.getAccessScheme().size(); i++) {
            for (int j = i + 1; j < ((ArrayData) exp.getOutputElement()).getShape().size(); j++) {
                builder.append(((ArrayData) exp.getOutputElement()).getShape().get(j));
                builder.append("LL * ");
            }
            builder.append("(");
            builder.append(printOperationExpression(exp.getAccessScheme().get(i)));
            builder.append(")");
            if (i != exp.getAccessScheme().size() - 1) {
                builder.append(" + ");
            }
        }

        if (!exp.getAccessScheme().isEmpty()) {
            builder.append("]");
        }

        return builder;
    }

    /**
     * Adds a KV-Pair to the inlining values
     * @param parameter
     * @param expression
     */
    public static void addParallelVariableInlining(Data parameter, OperationExpression expression) {
        parallelVariableInlining.put(parameter, expression);
    }

    /**
     * Removes a KV-Pair from the inlining values
     * @param parameter
     */
    public static void removeParallelVariableInlining(Data parameter) {
        parallelVariableInlining.remove(parameter);
    }

    /**
     * Removes a KV-Pair from the inlining values
     * @param parameter
     */
    public static OperationExpression getParallelVariableInlining(Data parameter) {
        return parallelVariableInlining.get(parameter);
    }

}