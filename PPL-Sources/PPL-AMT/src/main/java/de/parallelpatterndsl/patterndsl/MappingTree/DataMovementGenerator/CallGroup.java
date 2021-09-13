package de.parallelpatterndsl.patterndsl.MappingTree.DataMovementGenerator;

import de.parallelpatterndsl.patterndsl.MappingTree.Nodes.DataControl.DataPlacement;
import de.parallelpatterndsl.patterndsl.MappingTree.Nodes.Function.MainMapping;
import de.parallelpatterndsl.patterndsl.MappingTree.Nodes.ParallelCalls.FusedParallelCallMapping;
import de.parallelpatterndsl.patterndsl.MappingTree.Nodes.ParallelCalls.ParallelCallMapping;
import de.parallelpatterndsl.patterndsl.MappingTree.Nodes.Plain.SimpleExpressionBlockMapping;
import de.parallelpatterndsl.patterndsl.PatternTypes;
import de.parallelpatterndsl.patterndsl.abstractPatternTree.DataElements.ArrayData;
import de.parallelpatterndsl.patterndsl.abstractPatternTree.DataElements.Data;
import de.parallelpatterndsl.patterndsl.abstractPatternTree.DataElements.DataAccess.DataAccess;
import de.parallelpatterndsl.patterndsl.abstractPatternTree.DataElements.PrimitiveData;
import de.parallelpatterndsl.patterndsl.abstractPatternTree.Nodes.Plain.SimpleExpressionBlockNode;
import de.parallelpatterndsl.patterndsl.expressions.AssignmentExpression;
import de.parallelpatterndsl.patterndsl.expressions.IRLExpression;
import de.parallelpatterndsl.patterndsl.expressions.OperationExpression;
import de.parallelpatterndsl.patterndsl.expressions.Operator;
import de.parallelpatterndsl.patterndsl.hardwareDescription.Processor;
import de.parallelpatterndsl.patterndsl.helperLibrary.RandomStringGenerator;

import java.util.*;

public class CallGroup extends ParallelGroup{
    private MainMapping mainFunction;

    private ArrayList<ParallelCallMapping> group;

    private int remaining;

    private Optional<SimpleExpressionBlockMapping> parameterReplacementExpressions;

    private Optional<SimpleExpressionBlockMapping> resultReplacementExpression;

    private AssignmentExpression definition;

    private boolean isFirstAccess;

    private HashMap<Data, ArrayList<DataPlacement>>  inputSet;

    private String identifier;

    private HashSet<Processor> processors;

    public CallGroup(MainMapping mainFunction, ArrayList<ParallelCallMapping> group) {
        this.mainFunction = mainFunction;
        this.group = group;
        this.remaining = group.size();
        this.isFirstAccess = true;
        inputSet = new HashMap<>();
        this.identifier = RandomStringGenerator.getAlphaNumericString();

        createInputParameterAssignments();
        createOutputParameterAssignment();
        updateNewNodes();
    }

    public boolean isFirstAccess() {
        return isFirstAccess;
    }

    public void setFirstAccess(boolean firstAccess) {
        isFirstAccess = firstAccess;
    }

    public MainMapping getMainFunction() {
        return mainFunction;
    }

    public ArrayList<ParallelCallMapping> getGroup() {
        return group;
    }

    public int getRemaining() {
        return remaining;
    }

    public Optional<SimpleExpressionBlockMapping> getParameterReplacementExpressions() {
        return parameterReplacementExpressions;
    }

    public Optional<SimpleExpressionBlockMapping> getResultReplacementExpression() {
        return resultReplacementExpression;
    }

    public boolean isLastCall() {
        if (remaining == 1) {
            return true;
        } else {
            remaining --;
            return  false;
        }
    }

    @Override
    public HashMap<Data, ArrayList<DataPlacement>> getFullInputPlacement() {
        if (inputSet.isEmpty()) {
            HashMap<Data, ArrayList<DataPlacement>> result = new HashMap<>();

            for (ParallelCallMapping call : group) {
                for (DataPlacement placement : call.getNecessaryData()) {
                    if (result.containsKey(placement.getDataElement())) {
                        result.get(placement.getDataElement()).add(placement);
                    } else {
                        ArrayList<DataPlacement> partial = new ArrayList<>();
                        partial.add(placement);
                        result.put(placement.getDataElement(), partial);
                    }
                }
            }
            inputSet = result;
        }

        return inputSet;
    }

    @Override
    public void resetRemaining() {
        remaining = group.size();
    }

    @Override
    public String getGroupIdentifier() {
        return identifier;
    }

    @Override
    public Set<Processor> getProcessors() {
        if (processors == null) {
            processors = new HashSet<>();
            for (ParallelCallMapping mapping: group ) {
                processors.add(mapping.getExecutor());
            }
        }
        return processors;
    }


    /**
     * replaces complex parameters with single new variables and creates/returns the corresponding initialization as a simple expression block.
     */
    private void createInputParameterAssignments () {
        ParallelCallMapping call = group.get(0);
        ArrayList<IRLExpression> parameterAssignments = new ArrayList<>();
        ArrayList<Data> callOperands = new ArrayList<>();
        ArrayList<Operator> callOperators = new ArrayList<>();
        callOperators.add(Operator.LEFT_CALL_PARENTHESIS);
        callOperands.add(((AssignmentExpression) call.getDefinition().getExpression()).getRhsExpression().getOperands().get(0));

        definition = new AssignmentExpression(((AssignmentExpression) call.getDefinition().getExpression()).getOutputElement(), ((AssignmentExpression) call.getDefinition().getExpression()).getAccessScheme(),((AssignmentExpression) call.getDefinition().getExpression()).getRhsExpression(),((AssignmentExpression) call.getDefinition().getExpression()).getOperator());

        for (OperationExpression parameter: call.getArgumentExpressions()) {
            if (!parameter.getOperators().isEmpty()) {
                String randIndex = RandomStringGenerator.getAlphaNumericString();
                Data replacer;
                ArrayList<Integer> shape = parameter.getShape();

                if (shape.isEmpty()) {
                    replacer = new PrimitiveData("dataMovementReplacer_" + randIndex, parameter.getOperands().get(0).getTypeName(), false);
                } else {
                    replacer = new ArrayData("dataMovementReplacer_" + randIndex, parameter.getOperands().get(0).getTypeName(), false, shape, false);
                    ((ArrayData) replacer).setLocalPointer(true);
                }

                callOperands.add(replacer);
                mainFunction.getVariableTable().put(replacer.getIdentifier(), replacer);
                call.getVariableTable().put(replacer.getIdentifier(), replacer);

                AssignmentExpression initReplacer = new AssignmentExpression(replacer, new ArrayList<>(), new OperationExpression(parameter.getOperands(), parameter.getOperators()), Operator.ASSIGNMENT);
                parameterAssignments.add(initReplacer);

            } else {
                callOperands.addAll(parameter.getOperands());
            }

            callOperators.add(Operator.COMMA);
        }
        if (callOperators.size() > 1) {
            callOperators.remove(callOperators.size() - 1);
        }
        callOperators.add(Operator.RIGHT_CALL_PARENTHESIS);

        OperationExpression parameterReplacement = new OperationExpression(callOperands,callOperators);

        definition.setRhsExpression(parameterReplacement);

        if (parameterAssignments.isEmpty()) {
            parameterReplacementExpressions = Optional.empty();
        } else {
            parameterReplacementExpressions = Optional.of(new SimpleExpressionBlockMapping(Optional.of(mainFunction), call.getVariableTable(), new SimpleExpressionBlockNode(parameterAssignments)));
        }


    }


    /**
     * Replaces a complex output access with a single data element and return the reassignment of the original value.
     */
    private void createOutputParameterAssignment () {
        ParallelCallMapping call = group.get(0);
        AssignmentExpression assignmentExpression = definition;

        if (!assignmentExpression.getAccessScheme().isEmpty()) {
            AssignmentExpression outputDefinition = new AssignmentExpression(assignmentExpression.getOutputElement(), assignmentExpression.getAccessScheme(), assignmentExpression.getRhsExpression(), assignmentExpression.getOperator());

            ArrayList<Integer> shape = outputDefinition.getShape();
            String randIndex = RandomStringGenerator.getAlphaNumericString();
            Data replacer;

            if (shape.isEmpty()) {
                replacer = new PrimitiveData("dataMovementReplacer_" + randIndex, outputDefinition.getOutputElement().getTypeName(), false);
            } else {
                replacer = new ArrayData("dataMovementReplacer_" + randIndex, outputDefinition.getOutputElement().getTypeName(), false, shape, false);
                ((ArrayData) replacer).setLocalPointer(true);
            }

            mainFunction.getVariableTable().put(replacer.getIdentifier(), replacer);
            call.getVariableTable().put(replacer.getIdentifier(), replacer);

            ArrayList<Data> callOperands = new ArrayList<>();
            callOperands.add(replacer);

            AssignmentExpression replacement = new AssignmentExpression(outputDefinition.getOutputElement(), outputDefinition.getAccessScheme(),new OperationExpression(callOperands, new ArrayList<>()), Operator.ASSIGNMENT);

            ArrayList<IRLExpression> result = new ArrayList<>();

            outputDefinition.setOutputElement(replacer);
            outputDefinition.setAccessScheme(new ArrayList<>());

            result.add(replacement);

            resultReplacementExpression = Optional.of(new SimpleExpressionBlockMapping(Optional.of(mainFunction), call.getVariableTable(), new SimpleExpressionBlockNode(result)));

            definition = outputDefinition;

            for (ParallelCallMapping node : group ) {
                node.getDefinition().setExpression(definition);
            }

        } else {

            resultReplacementExpression = Optional.empty();
        }
    }


    /**
     * creates the input and output accesses for the new serial block nodes nodes.
     * @param mapping
     */
    private void updateDataAccesses(SimpleExpressionBlockMapping mapping) {
            ArrayList<Data> inputData = new ArrayList<>();
            ArrayList<Data> outputData = new ArrayList<>();
            ArrayList<DataAccess> inputAccess = new ArrayList<>();
            ArrayList<DataAccess> outputAccess = new ArrayList<>();

            for (IRLExpression expression: mapping.getExpressionList() ) {
                ArrayList<DataAccess> accesses = expression.getDataAccesses(PatternTypes.SEQUENTIAL);

                for (DataAccess dataAccess: accesses ) {
                    if (dataAccess.isReadAccess()) {
                        inputAccess.add(dataAccess);
                        if (!inputData.contains(dataAccess.getData())) {
                            inputData.add(dataAccess.getData());
                        }
                    } else {
                        outputAccess.add(dataAccess);
                        if (!outputData.contains(dataAccess.getData())) {
                            outputData.add(dataAccess.getData());
                        }
                    }
                }
            }
            mapping.setInputAccesses(inputAccess);
            mapping.setInputElements(inputData);
            mapping.setOutputAccesses(outputAccess);
            mapping.setOutputElements(outputData);
    }

    /**
     * Update the the callGroup to contain data accesses.
     */
    private void updateNewNodes() {

        ArrayList<Data> inputData = new ArrayList<>();
        ArrayList<Data> outputData = new ArrayList<>();
        ArrayList<DataAccess> inputAccess = new ArrayList<>();
        ArrayList<DataAccess> outputAccess = new ArrayList<>();

        ArrayList<DataAccess> accesses = definition.getDataAccesses(PatternTypes.SEQUENTIAL);

        for (DataAccess dataAccess: accesses ) {
            if (dataAccess.isReadAccess()) {
                inputAccess.add(dataAccess);
                if (!inputData.contains(dataAccess.getData())) {
                    inputData.add(dataAccess.getData());
                }
            } else {
                outputAccess.add(dataAccess);
                if (!outputData.contains(dataAccess.getData())) {
                    outputData.add(dataAccess.getData());
                }
            }
        }


        for (ParallelCallMapping member: group ) {
            member.setInputAccesses(inputAccess);
            member.setInputElements(inputData);
            member.setOutputAccesses(outputAccess);
            member.setOutputElements(outputData);
        }

        parameterReplacementExpressions.ifPresent(this::updateDataAccesses);
        resultReplacementExpression.ifPresent(this::updateDataAccesses);
    }
}
