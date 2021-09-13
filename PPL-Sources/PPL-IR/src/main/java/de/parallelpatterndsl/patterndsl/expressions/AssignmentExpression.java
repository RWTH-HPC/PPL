package de.parallelpatterndsl.patterndsl.expressions;

import de.parallelpatterndsl.patterndsl.PatternTypes;
import de.parallelpatterndsl.patterndsl.abstractPatternTree.DataElements.ArrayData;
import de.parallelpatterndsl.patterndsl.abstractPatternTree.DataElements.Data;
import de.parallelpatterndsl.patterndsl.abstractPatternTree.DataElements.DataAccess.*;
import de.parallelpatterndsl.patterndsl.abstractPatternTree.DataElements.LiteralData;
import de.parallelpatterndsl.patterndsl.abstractPatternTree.DataElements.PrimitiveData;

import java.util.ArrayList;
import java.util.HashMap;

/**
 * A class that stores the assignment of a variable to a new value.
 */
public class AssignmentExpression extends IRLExpression {

    /**
     * The element which is changed within the assignment.
     */
    private Data outputElement;

    /**
     * The index access schemes when assigning individual values in an array.
     */
    private ArrayList<OperationExpression> accessScheme;

    /**
     * The Operation expression defining the new value.
     */
    private OperationExpression rhsExpression;

    /**
     * The Operator for assigning the new value e.g. "=" or "+=".
     */
    private final Operator operator;

    public AssignmentExpression(Data outputElement, ArrayList<OperationExpression> accessScheme, OperationExpression rhsExpression, Operator operator) {
        this.outputElement = outputElement;
        this.accessScheme = accessScheme;
        this.rhsExpression = rhsExpression;
        this.operator = operator;
    }

    public Data getOutputElement() {
        return outputElement;
    }

    public ArrayList<OperationExpression> getAccessScheme() {
        return accessScheme;
    }

    public OperationExpression getRhsExpression() {
        return rhsExpression;
    }

    public Operator getOperator() {
        return operator;
    }

    public void setOutputElement(Data outputElement) {
        this.outputElement = outputElement;
    }

    public void setAccessScheme(ArrayList<OperationExpression> accessScheme) {
        this.accessScheme = accessScheme;
    }

    public void setRhsExpression(OperationExpression rhsExpression) {
        this.rhsExpression = rhsExpression;
    }

    @Override
    public ArrayList<DataAccess> getDataAccesses(PatternTypes patternType) {
        ArrayList<DataAccess> result = new ArrayList<>();
        // handle special IO accesses
        if (rhsExpression.isHasIOData()) {
            result.add(new IODataAccess(outputElement,false));
        } else {
            if (outputElement instanceof PrimitiveData) {
                if (patternType == PatternTypes.REDUCE && outputElement.isReturnData()) {
                    result.add(new ReduceDataAccess(outputElement, false));
                } else {
                    result.add(new DataAccess(outputElement, false));
                }
            } else if (outputElement instanceof ArrayData) {

                if (patternType == PatternTypes.SEQUENTIAL || accessScheme.isEmpty() || !outputElement.isParameter()) {
                    result.add(new DataAccess(outputElement, false));
                } else if (patternType == PatternTypes.MAP && !accessScheme.isEmpty()) {
                    OperationExpression accessPattern = accessScheme.get(0);

                    // Handle the different possible linear access schemes
                    if (accessPattern.getOperators().size() == 0) {
                        result.add(new MapDataAccess(outputElement,false, 1, 0 ));
                    } else if (accessPattern.getOperators().size() == 1) {
                        if (accessPattern.getOperators().get(0) == Operator.MULTIPLICATION) {
                            result.add(new MapDataAccess(outputElement,false, ((LiteralData<Integer>) accessPattern.getOperands().get(0)).getValue(), 0 ));
                        } else if (accessPattern.getOperators().get(0) == Operator.PLUS) {
                            result.add(new MapDataAccess(outputElement,false, 1, ((LiteralData<Integer>) accessPattern.getOperands().get(1)).getValue() ));
                        } else if (accessPattern.getOperators().get(0) == Operator.MINUS) {
                            result.add(new MapDataAccess(outputElement,false, 1, ((LiteralData<Integer>) accessPattern.getOperands().get(1)).getValue() * -1));
                        }
                    } else if (accessPattern.getOperators().size() == 2) {
                        int normalizationFactor = 1;
                        if (accessPattern.getOperators().get(1) == Operator.MINUS) {
                            normalizationFactor = -1;
                        }
                        result.add(new MapDataAccess(outputElement,false, ((LiteralData<Integer>) accessPattern.getOperands().get(0)).getValue(), ((LiteralData<Integer>) accessPattern.getOperands().get(2)).getValue() * normalizationFactor));
                    }
                } else if (patternType == PatternTypes.STENCIL && !accessScheme.isEmpty()) {
                    ArrayList<Integer> scalingFactors = new ArrayList<>();
                    ArrayList<Integer> shiftOffsets = new ArrayList<>();
                    ArrayList<String> ruleBaseIndex = new ArrayList<>();

                    for (OperationExpression accessPattern : accessScheme) {
                        // Handle the different possible linear access schemes
                        if (accessPattern.getOperators().size() == 0) {
                            scalingFactors.add(1);
                            ruleBaseIndex.add(accessPattern.getOperands().get(0).getIdentifier());
                            shiftOffsets.add(0);
                        } else if (accessPattern.getOperators().size() == 1) {
                            if (accessPattern.getOperators().get(0) == Operator.MULTIPLICATION) {
                                scalingFactors.add(((LiteralData<Integer>) accessPattern.getOperands().get(0)).getValue());
                                shiftOffsets.add(0);
                                ruleBaseIndex.add(accessPattern.getOperands().get(1).getIdentifier());
                            } else if (accessPattern.getOperators().get(0) == Operator.PLUS) {
                                scalingFactors.add(1);
                                shiftOffsets.add(((LiteralData<Integer>) accessPattern.getOperands().get(1)).getValue());
                                ruleBaseIndex.add(accessPattern.getOperands().get(0).getIdentifier());
                            }else if (accessPattern.getOperators().get(0) == Operator.MINUS) {
                                scalingFactors.add(1);
                                shiftOffsets.add(((LiteralData<Integer>) accessPattern.getOperands().get(1)).getValue() * -1);
                                ruleBaseIndex.add(accessPattern.getOperands().get(0).getIdentifier());
                            }
                        } else if (accessPattern.getOperators().size() == 2) {
                            int normalizationFactor = 1;
                            if (accessPattern.getOperators().get(1) == Operator.MINUS) {
                                normalizationFactor = -1;
                            }
                            scalingFactors.add(((LiteralData<Integer>) accessPattern.getOperands().get(0)).getValue());
                            shiftOffsets.add(((LiteralData<Integer>) accessPattern.getOperands().get(2)).getValue() * normalizationFactor);
                            ruleBaseIndex.add(accessPattern.getOperands().get(1).getIdentifier());
                        }
                    }
                    result.add(new StencilDataAccess(outputElement,false, scalingFactors, shiftOffsets, ruleBaseIndex));
                } else if (patternType == PatternTypes.DYNAMIC_PROGRAMMING && !accessScheme.isEmpty()) {
                    ArrayList<Integer> shiftOffsets = new ArrayList<>();
                    ArrayList<String> ruleBaseIndex = new ArrayList<>();

                    for (OperationExpression accessPattern : accessScheme) {
                        // Handle the different possible linear access schemes
                        if (accessPattern.getOperators().size() == 0) {
                            Data element = accessPattern.getOperands().get(0);
                            if (element instanceof LiteralData) {
                                ruleBaseIndex.add("");
                                shiftOffsets.add(((LiteralData<Integer>) element).getValue());
                            } else if (element instanceof PrimitiveData) {
                                ruleBaseIndex.add(element.getIdentifier());
                                shiftOffsets.add(0);
                            }
                        } else if (accessPattern.getOperators().size() == 1) {
                            if (accessPattern.getOperators().get(0) == Operator.PLUS) {
                                shiftOffsets.add(((LiteralData<Integer>) accessPattern.getOperands().get(1)).getValue());
                                ruleBaseIndex.add(accessPattern.getOperands().get(0).getIdentifier());
                            }else if (accessPattern.getOperators().get(0) == Operator.MINUS) {
                                shiftOffsets.add(((LiteralData<Integer>) accessPattern.getOperands().get(1)).getValue() * -1);
                                ruleBaseIndex.add(accessPattern.getOperands().get(0).getIdentifier());
                            }
                        }
                    }
                    result.add(new DynamicProgrammingDataAccess(outputElement,false, shiftOffsets, ruleBaseIndex));
                } else if (patternType == PatternTypes.RECURSION && !accessScheme.isEmpty()) {
                    //TODO!!!
                }
            }
            result.addAll(getRhsExpression().getDataAccesses( patternType));
        }
        return result;
    }

    @Override
    public int getOperationCount() {
        if (operator == Operator.ASSIGNMENT) {
            return rhsExpression.getOperationCount();
        } else {
            return rhsExpression.getOperationCount() + 1;
        }
    }

    @Override
    public ArrayList<Integer> getShape() {
        ArrayList<Integer> result = new ArrayList<>();
        if (outputElement instanceof ArrayData) {
            result = (ArrayList<Integer>) ((ArrayData) outputElement).getShape().clone();
            for (int i = 0; i < accessScheme.size(); i++) {
                result.remove(0);
            }
        }
        return result;
    }

    @Override
    public boolean isHasIOData() {
        return rhsExpression.isHasIOData();
    }

    @Override
    public AssignmentExpression createInlineCopy(ArrayList<Data> globalVars, String inlineIdentifier, HashMap<String, Data> variableTable) {
        Data newOutput = variableTable.get(outputElement.getIdentifier()+ "_" + inlineIdentifier);

        ArrayList<OperationExpression> newAccessScheme = new ArrayList<>();

        for (OperationExpression old: accessScheme ) {
            newAccessScheme.add(old.createInlineCopy(globalVars, inlineIdentifier, variableTable));
        }

        OperationExpression newRHS = rhsExpression.createInlineCopy(globalVars, inlineIdentifier, variableTable);

        return new AssignmentExpression(newOutput, newAccessScheme, newRHS, operator);
    }

    @Override
    public void replaceDataElement(Data oldData, Data newData) {
        if (outputElement == oldData) {
            outputElement = newData;
        } else {
            rhsExpression.replaceDataElement(oldData, newData);
        }
    }
}
