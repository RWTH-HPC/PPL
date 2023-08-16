package de.parallelpatterndsl.patterndsl.expressions;

import de.parallelpatterndsl.patterndsl.PatternTypes;
import de.parallelpatterndsl.patterndsl.Preprocessing.VariableReplacementStack;
import de.parallelpatterndsl.patterndsl.abstractPatternTree.DataElements.*;
import de.parallelpatterndsl.patterndsl.abstractPatternTree.DataElements.DataAccess.*;
import de.parallelpatterndsl.patterndsl.helperLibrary.DeepCopyHelper;
import de.parallelpatterndsl.patterndsl.helperLibrary.PredefinedFunctions;
import de.se_rwth.commons.logging.Log;

import java.util.ArrayList;
import java.util.HashMap;

/**
 * A class that stores a sequence of operations as a list of operands and operators.
 */
public class OperationExpression extends IRLExpression{

    /**
     * The list of data elements in the expression.
     */
    private final ArrayList<Data> operands;

    /**
     * The list of operators in the expression.
     */
    private final ArrayList<Operator> operators;

    /**
     * True, iff the expression contains IO expressions.
     */
    private boolean hasIOData;

    public OperationExpression(ArrayList<Data> operands, ArrayList<Operator> operators) {
        this.operands = operands;
        this.operators = operators;
        hasIOData = false;
        for (Data data: operands ) {
            if (data instanceof IOData) {
                hasIOData = true;
                break;
            }
        }
    }

    public ArrayList<Data> getOperands() {
        return operands;
    }

    public ArrayList<Operator> getOperators() {
        return operators;
    }

    public boolean isHasIOData() {
        return hasIOData;
    }

    @Override
    public OperationExpression createInlineCopy(ArrayList<Data> globalVars, String inlineIdentifier, HashMap<String, Data> variableTable) {
        ArrayList<Operator> newOperators = new ArrayList<>(getOperators());
        ArrayList<Data> newOperands = new ArrayList<>();

        for (Data operand: operands ) {
            if (globalVars.contains(operand)) {
                newOperands.add(operand);
            } else if (operand instanceof FunctionInlineData) {
                FunctionInlineData newData;
                if (operand.getIdentifier().startsWith("inlineFunctionValue") && variableTable.containsKey(operand.getIdentifier() + "_" + operand.getIdentifier())) {
                    newData = (FunctionInlineData) variableTable.get(operand.getIdentifier());
                } else {
                    newData = (FunctionInlineData) operand;
                }
                ((FunctionInlineData) newData).createInlineCopies(globalVars, inlineIdentifier, variableTable);
                newData.setInlineEnding(((FunctionInlineData) operand).getInlineEnding());
                newOperands.add(newData);
            } else if (operand instanceof LiteralData){
                newOperands.add(operand.createInlineCopy(inlineIdentifier));
            } else if (PredefinedFunctions.contains(operand.getIdentifier())){
                newOperands.add(operand.createInlineCopy(operand.getIdentifier()));
            } else if (operand instanceof FunctionReturnData){
                newOperands.add(operand.createInlineCopy(operand.getIdentifier()));
            } else if (operand instanceof IOData){
                newOperands.add(operand.createInlineCopy(operand.getIdentifier()));
            } else {
                if (VariableReplacementStack.getCurrentTable().containsKey(operand)) {
                    newOperands.add(VariableReplacementStack.getCurrentTable().get(operand));
                } else if (variableTable.containsValue(operand)){
                    newOperands.add(operand);
                } else {
                    VariableReplacementStack.getCurrentTable();
                    Log.error("Variable: " + operand.getIdentifier() + " unknown.");
                }
            }
        }

        return new OperationExpression(newOperands, newOperators);
    }

    @Override
    public void replaceDataElement(Data oldData, Data newData) {
        for (int i = 0; i < operands.size(); i++) {
            Data operand = operands.get(i);
            if (operand == oldData) {
                operands.remove(i);
                operands.add(i, newData);
            } else if (operand instanceof FunctionInlineData) {
                ((FunctionInlineData) operand).replaceDataElement(oldData, newData);
            }
        }
    }

    @Override
    public int getLoadStores() {
        int sum = 0;
        for (Data operand: operands) {
            if (operand instanceof PrimitiveData || operand instanceof ArrayData) {
                sum ++;
            }
        }
        return sum;
    }

    @Override
    public OperationExpression deepCopy() {

        ArrayList<Operator> newOperators = new ArrayList<>(operators);

        ArrayList<Data> newOperands = new ArrayList<>();

        for (Data data: operands ) {
            if (data instanceof PrimitiveData || data instanceof ArrayData || data instanceof FunctionInlineData) {
                if (data.getIdentifier().equals("Element_Exists_In_Vector") || data.getIdentifier().equals("Get_Size_Of_Vector")) {
                    newOperands.add(data.deepCopy());
                } else {
                    newOperands.add(DeepCopyHelper.currentScope().get(data.getIdentifier()));
                }
            } else {
                newOperands.add(data.deepCopy());
            }
        }

        return new OperationExpression(newOperands, newOperators);
    }

    @Override
    public ArrayList<DataAccess> getDataAccesses(PatternTypes patternType) {
        ArrayList<DataAccess> result = new ArrayList<>();

        // handle special IO accesses
        if (hasIOData) {
            for (Data dataElement: operands ) {
                if (dataElement instanceof PrimitiveData || dataElement instanceof ArrayData) {
                    result.add(new IODataAccess(dataElement, true));
                } else if (dataElement instanceof FunctionInlineData) {
                    result.addAll(((FunctionInlineData) dataElement).getCall().getDataAccesses(patternType));
                    result.add(new DataAccess(dataElement, false));
                    result.add(new IODataAccess(dataElement, true));
                }
            }
        } else {
            for (int i = 0; i < operands.size(); i++) {
                Data dataElement = operands.get(i);
                if (dataElement instanceof PrimitiveData) {
                    result.add(new DataAccess(dataElement, true));
                } else if (dataElement instanceof ArrayData) {
                    if (patternType == PatternTypes.SEQUENTIAL || !dataElement.isParameter()) {
                        result.add(new DataAccess(dataElement, true));
                    } else {

                        int rightContext = getRightOperandContextIndex(i);
                        if (rightContext != -1) {
                            if (operators.get(rightContext) == Operator.LEFT_ARRAY_ACCESS) {
                                if (patternType == PatternTypes.MAP || patternType == PatternTypes.REDUCE) {
                                    // Test the structure W * INDEX + b. If W = 1 or b = 0 they can be left out
                                    if (operators.get(rightContext + 1) == Operator.RIGHT_ARRAY_ACCESS) {
                                        result.add(new MapDataAccess(dataElement, true, 1, 0));
                                    } else if (operators.get(rightContext + 1) == Operator.MULTIPLICATION) {
                                        if (operators.get(rightContext + 2) == Operator.RIGHT_ARRAY_ACCESS) {
                                            result.add(new MapDataAccess(dataElement, true, ((LiteralData<Integer>) operands.get(i+1)).getValue(), 0));
                                        }  else if (operators.get(rightContext + 2) == Operator.PLUS) {
                                            result.add(new MapDataAccess(dataElement, true, ((LiteralData<Integer>) operands.get(i+1)).getValue(), ((LiteralData<Integer>) operands.get(i+3)).getValue()));
                                        }  else if (operators.get(rightContext + 2) == Operator.MINUS) {
                                            result.add(new MapDataAccess(dataElement, true, ((LiteralData<Integer>) operands.get(i+1)).getValue(), ((LiteralData<Integer>) operands.get(i+3)).getValue() * -1));
                                        }
                                    }  else if (operators.get(rightContext + 1) == Operator.PLUS) {
                                        result.add(new MapDataAccess(dataElement, true, 1, ((LiteralData<Integer>) operands.get(i+2)).getValue()));
                                    }  else if (operators.get(rightContext + 1) == Operator.MINUS) {
                                        result.add(new MapDataAccess(dataElement, true, 1, ((LiteralData<Integer>) operands.get(i+2)).getValue() * -1));
                                    }
                                } else if (patternType == PatternTypes.STENCIL) {
                                    ArrayList<Integer> scalingFactors = new ArrayList<>();
                                    ArrayList<Integer> shiftOffsets = new ArrayList<>();
                                    ArrayList<String> ruleBaseIndex = new ArrayList<>();

                                    boolean canContinue = true;

                                    int rightHelperContext = rightContext;
                                    int currentOperand = i;

                                    while(canContinue) {
                                        // Test the structure W * INDEX + b. If W = 1 or b = 0 they can be left out
                                        if (operators.get(rightHelperContext + 1) == Operator.RIGHT_ARRAY_ACCESS) {
                                            scalingFactors.add(1);
                                            shiftOffsets.add(0);
                                            ruleBaseIndex.add(operands.get(currentOperand + 1).getIdentifier());

                                            if (rightHelperContext + 1 == operators.size() - 1) {
                                                canContinue = false;
                                            } else if (operators.get(rightHelperContext + 2) != Operator.LEFT_ARRAY_ACCESS) {
                                                canContinue = false;
                                            } else {
                                                currentOperand += 1;
                                                rightHelperContext += 2;
                                            }
                                        } else if (operators.get(rightHelperContext + 1) == Operator.MULTIPLICATION) {
                                            if (operators.get(rightHelperContext + 2) == Operator.RIGHT_ARRAY_ACCESS) {
                                                scalingFactors.add(((LiteralData<Integer>) operands.get(currentOperand+1)).getValue());
                                                shiftOffsets.add(0);
                                                ruleBaseIndex.add(operands.get(currentOperand + 2).getIdentifier());

                                                if (rightHelperContext + 2 == operators.size() - 1) {
                                                    canContinue = false;
                                                } else if (operators.get(rightHelperContext + 3) != Operator.LEFT_ARRAY_ACCESS) {
                                                    canContinue = false;
                                                } else {
                                                    currentOperand += 2;
                                                    rightHelperContext += 3;
                                                }
                                            }  else if (operators.get(rightContext + 2) == Operator.PLUS) {
                                                scalingFactors.add(((LiteralData<Integer>) operands.get(currentOperand+1)).getValue());
                                                shiftOffsets.add(((LiteralData<Integer>) operands.get(currentOperand+3)).getValue());
                                                ruleBaseIndex.add(operands.get(currentOperand + 2).getIdentifier());

                                                if (rightHelperContext + 3 == operators.size() - 1) {
                                                    canContinue = false;
                                                } else if (operators.get(rightHelperContext + 4) != Operator.LEFT_ARRAY_ACCESS) {
                                                    canContinue = false;
                                                } else {
                                                    currentOperand += 3;
                                                    rightHelperContext += 4;
                                                }
                                            }  else if (operators.get(rightContext + 2) == Operator.MINUS) {
                                                scalingFactors.add(((LiteralData<Integer>) operands.get(currentOperand+1)).getValue());
                                                shiftOffsets.add(((LiteralData<Integer>) operands.get(currentOperand+3)).getValue() * -1);
                                                ruleBaseIndex.add(operands.get(currentOperand + 2).getIdentifier());

                                                if (rightHelperContext + 3 == operators.size() - 1) {
                                                    canContinue = false;
                                                } else if (operators.get(rightHelperContext + 4) != Operator.LEFT_ARRAY_ACCESS) {
                                                    canContinue = false;
                                                } else {
                                                    currentOperand += 3;
                                                    rightHelperContext += 4;
                                                }
                                            }
                                        }  else if (operators.get(rightHelperContext + 1) == Operator.PLUS) {
                                            scalingFactors.add(1);
                                            shiftOffsets.add(((LiteralData<Integer>) operands.get(currentOperand+2)).getValue());
                                            ruleBaseIndex.add(operands.get(currentOperand + 1).getIdentifier());

                                            if (rightHelperContext + 2 == operators.size() - 1) {
                                                canContinue = false;
                                            } else if (operators.get(rightHelperContext + 3) != Operator.LEFT_ARRAY_ACCESS) {
                                                canContinue = false;
                                            } else {
                                                currentOperand += 2;
                                                rightHelperContext += 3;
                                            }
                                        }  else if (operators.get(rightHelperContext + 1) == Operator.MINUS) {
                                            scalingFactors.add(1);
                                            shiftOffsets.add(((LiteralData<Integer>) operands.get(currentOperand+2)).getValue() * -1);
                                            ruleBaseIndex.add(operands.get(currentOperand + 1).getIdentifier());

                                            if (rightHelperContext + 2 == operators.size() - 1) {
                                                canContinue = false;
                                            } else if (operators.get(rightHelperContext + 3) != Operator.LEFT_ARRAY_ACCESS) {
                                                canContinue = false;
                                            } else {
                                                currentOperand += 2;
                                                rightHelperContext += 3;
                                            }
                                        } else {
                                            canContinue = false;
                                        }

                                    }
                                    result.add(new StencilDataAccess(dataElement,true,scalingFactors,shiftOffsets,ruleBaseIndex));
                                } else if (patternType == PatternTypes.DYNAMIC_PROGRAMMING) {
                                    ArrayList<Integer> shiftOffsets = new ArrayList<>();
                                    ArrayList<String> ruleBaseIndex = new ArrayList<>();

                                    boolean canContinue = true;

                                    int rightHelperContext = rightContext;
                                    int currentOperand = i;

                                    while(canContinue) {
                                        // Test the structure INDEX + b. If b = 0 it can be left out
                                        if (operators.get(rightHelperContext + 1) == Operator.RIGHT_ARRAY_ACCESS) {
                                            if (operands.get(currentOperand + 1) instanceof LiteralData) {
                                                shiftOffsets.add(((LiteralData<Integer>) operands.get(currentOperand+1)).getValue());
                                                ruleBaseIndex.add("");
                                            } else if (operands.get(currentOperand + 1) instanceof PrimitiveData) {
                                                shiftOffsets.add(0);
                                                ruleBaseIndex.add(operands.get(currentOperand + 1).getIdentifier());
                                            }

                                            if (rightHelperContext + 1 == operators.size() - 1) {
                                                canContinue = false;
                                            } else if (operators.get(rightHelperContext + 2) != Operator.LEFT_ARRAY_ACCESS) {
                                                canContinue = false;
                                            } else {
                                                currentOperand += 1;
                                                rightHelperContext += 2;
                                            }
                                        }  else if (operators.get(rightContext + 1) == Operator.PLUS) {
                                            shiftOffsets.add(((LiteralData<Integer>) operands.get(currentOperand+2)).getValue());
                                            ruleBaseIndex.add(operands.get(currentOperand + 1).getIdentifier());

                                            if (rightHelperContext + 2 == operators.size() - 1) {
                                                canContinue = false;
                                            } else if (operators.get(rightHelperContext + 3) != Operator.LEFT_ARRAY_ACCESS) {
                                                canContinue = false;
                                            } else {
                                                currentOperand += 2;
                                                rightHelperContext += 3;
                                            }
                                        }  else if (operators.get(rightContext + 1) == Operator.MINUS) {
                                            shiftOffsets.add(((LiteralData<Integer>) operands.get(currentOperand+2)).getValue() * -1);
                                            ruleBaseIndex.add(operands.get(currentOperand + 1).getIdentifier());

                                            if (rightHelperContext + 2 == operators.size() - 1) {
                                                canContinue = false;
                                            } else if (operators.get(rightHelperContext + 3) != Operator.LEFT_ARRAY_ACCESS) {
                                                canContinue = false;
                                            } else {
                                                currentOperand += 2;
                                                rightHelperContext += 3;
                                            }
                                        } else {
                                            canContinue = false;
                                        }

                                    }
                                    result.add(new DynamicProgrammingDataAccess(dataElement,true,shiftOffsets,ruleBaseIndex));
                                } else if (patternType == PatternTypes.RECURSION) {
                                    //TODO:
                                }
                            } else {
                                result.add(new DataAccess(dataElement, true));
                            }
                        } else {
                            result.add(new DataAccess(dataElement, true));
                        }


                    }


                } else if (dataElement instanceof FunctionInlineData) {
                    result.addAll(((FunctionInlineData) dataElement).getCall().getDataAccesses(patternType));
                    result.add(new DataAccess(dataElement, false));
                    result.add(new DataAccess(dataElement, true));
                }
            }
        }

        return result;
    }

    @Override
    public boolean hasProfilingInfo() {
        return operands.stream().filter(x -> x instanceof FunctionInlineData).anyMatch(x -> ((FunctionInlineData) x).getCall().getOperands().stream().anyMatch(y -> (y instanceof FunctionReturnData && (y.getIdentifier().startsWith("get_time") || y.getIdentifier().startsWith("get_maxRusage")))));
    }

    @Override
    public boolean hasExit() {
        return operands.stream().filter(x -> x instanceof FunctionInlineData).anyMatch(x -> ((FunctionInlineData) x).getCall().getOperands().stream().anyMatch(y -> (y instanceof FunctionReturnData && y.getIdentifier().equals("exit"))));
    }

    @Override
    public int getOperationCount() {
        int result = 0;

        boolean isInIndexAccess = false;

        for (Operator operator: operators ) {
            if (operator == Operator.LEFT_ARRAY_ACCESS) {
                isInIndexAccess = true;
            }
            if (operator == Operator.RIGHT_ARRAY_ACCESS) {
                isInIndexAccess = false;
            }
            if (!isInIndexAccess) {
                result += Operator.getCount(operator);
            }
        }

        return result;
    }

    /**
     * Gets the index of the next operator for a given operand index.
     * @param operandIndex
     * @return -1 if no right context is available.
     */
    public int getRightOperandContextIndex(int operandIndex) {
        if (operators.isEmpty()) {
            return -1;
        }
        int result = 0;
        int currentOperand = 0;

        while ( operandIndex >= currentOperand) {
            if (result < operators.size() - 1) {
                int operatorLookAhead = result + 1;

                // handle multiple array accesses
                if (operators.get(result) == Operator.RIGHT_ARRAY_ACCESS && operators.get(operatorLookAhead) == Operator.LEFT_ARRAY_ACCESS) {
                   if (operandIndex == currentOperand) {
                       return result;
                   } else {
                       result += 2;
                       currentOperand++;
                   }
                } else if (Operator.arity(operators.get(result)) == 2) { // handle operators with arity of 2
                    if (operandIndex == currentOperand) {
                        return result;
                    } else {
                        result++;
                        currentOperand++;
                    }
                } else if (Operator.arity(operators.get(result)) == 1) { // handle operators with arity of 1
                    if (operandIndex == currentOperand) {
                        if (operators.get(result) == Operator.RIGHT_ARRAY_ACCESS || operators.get(result) == Operator.RIGHT_ARRAY_DEFINITION || operators.get(result) == Operator.RIGHT_CALL_PARENTHESIS ||operators.get(result) == Operator.RIGHT_PARENTHESIS ) {
                            return result;
                        } else {
                            result++;
                        }
                    } else {
                        result++;
                    }
                }

            } else {
                if (Operator.arity(operators.get(result)) == 2) { // handle operators with arity of 2
                    if (operandIndex == currentOperand) {
                        return result;
                    } else {
                        return -1;
                    }
                } else if (Operator.arity(operators.get(result)) == 1) { // handle operators with arity of 1
                    if (operandIndex == currentOperand) {
                        if (operators.get(result) == Operator.RIGHT_ARRAY_ACCESS || operators.get(result) == Operator.RIGHT_ARRAY_DEFINITION || operators.get(result) == Operator.RIGHT_CALL_PARENTHESIS ||operators.get(result) == Operator.RIGHT_PARENTHESIS ) {
                            return result;
                        } else {
                            return -1;
                        }
                    } else {
                        Log.error("False Expression syntax!");
                        throw new RuntimeException("Critical error!");
                    }
                }
            }
        }

        return result;
    }


    @Override
    public ArrayList<Integer> getShape() {
        ArrayList<Integer> result = new ArrayList<>();

        for (int i = 0; i < operands.size(); i++) {
            if (!(operands.get(i) instanceof ArrayData || operands.get(i) instanceof FunctionInlineData)) {
                continue;
            }

            if (operands.get(i) instanceof FunctionInlineData) {
                FunctionInlineData data = (FunctionInlineData) operands.get(i);
                if (data.getShape().size() > 0) {
                    return data.getShape();
                }
            }

            int newContext = getRightOperandContextIndex(i);

            //counts the number of consecutive array accesses to a data element and generates the corresponding shape.
            //The current definition does not allow for nested index accesses.
            if (operands.get(i) instanceof ArrayData) {
                ArrayData data = (ArrayData) operands.get(i);
                int nextOperator = newContext;
                //handle single/last element
                if (newContext == -1) {
                    return data.getShape();
                }
                if (operators.get(newContext) == Operator.LEFT_ARRAY_ACCESS) {
                    int count = 1;
                    while (true) {
                        if (operators.size() - nextOperator < 2) {
                            break;
                        }
                        if (operators.size() - nextOperator > 2) {
                            if (operators.get(nextOperator + 1) == Operator.RIGHT_ARRAY_ACCESS && operators.get(nextOperator + 2) == Operator.LEFT_ARRAY_ACCESS) {
                                count++;
                                nextOperator++;
                            }
                        }
                        nextOperator++;
                    }

                    result = (ArrayList<Integer>) data.getShape().clone();

                    for (int j = 0; j < count; j++) {
                        result.remove(0);
                    }
                    return result;
                } else {
                    return data.getShape();
                }

            }
        }

        return result;
    }

    public OperationExpression simpleCopy() {
        return new OperationExpression(new ArrayList<>(operands), new ArrayList<>(operators));
    }
}
