package de.parallelpatterndsl.patterndsl.MappingTree.Nodes.ParallelCalls;

import de.parallelpatterndsl.patterndsl.MappingTree.AbstractMappingTree;
import de.parallelpatterndsl.patterndsl.MappingTree.DataMovementGenerator.ParallelGroup;
import de.parallelpatterndsl.patterndsl.MappingTree.Nodes.DataControl.BarrierMapping;
import de.parallelpatterndsl.patterndsl.MappingTree.Nodes.DataControl.DataMovementMapping;
import de.parallelpatterndsl.patterndsl.MappingTree.Nodes.DataControl.DataPlacement;
import de.parallelpatterndsl.patterndsl.MappingTree.Nodes.DataControl.EndPoint;
import de.parallelpatterndsl.patterndsl.MappingTree.Nodes.Function.ParallelMapping;
import de.parallelpatterndsl.patterndsl.MappingTree.Nodes.FunctionMapping;
import de.parallelpatterndsl.patterndsl.MappingTree.Nodes.MappingNode;
import de.parallelpatterndsl.patterndsl.MappingTree.Nodes.Plain.CallMapping;
import de.parallelpatterndsl.patterndsl.MappingTree.Nodes.Plain.ComplexExpressionMapping;
import de.parallelpatterndsl.patterndsl.MappingTree.SupportFunction;
import de.parallelpatterndsl.patterndsl.MappingTree.Visitor.AMTVisitor;
import de.parallelpatterndsl.patterndsl.abstractPatternTree.DataElements.ArrayData;
import de.parallelpatterndsl.patterndsl.abstractPatternTree.DataElements.Data;
import de.parallelpatterndsl.patterndsl.abstractPatternTree.DataElements.DataAccess.*;
import de.parallelpatterndsl.patterndsl.abstractPatternTree.DataElements.PrimitiveData;
import de.parallelpatterndsl.patterndsl.abstractPatternTree.Nodes.Plain.CallNode;
import de.parallelpatterndsl.patterndsl.expressions.AssignmentExpression;
import de.parallelpatterndsl.patterndsl.expressions.IRLExpression;
import de.parallelpatterndsl.patterndsl.expressions.OperationExpression;
import de.parallelpatterndsl.patterndsl.expressions.Operator;
import de.parallelpatterndsl.patterndsl.hardwareDescription.Processor;
import de.se_rwth.commons.logging.Log;

import java.util.*;

/**
 * Defines a simple parallel call on the CPU.
 */
public class ParallelCallMapping extends CallMapping {

    /**
     * Stores the first index for each dimension.
     */
    private ArrayList<Long> startIndex;

    /**
     * Stores the number of iterations for each dimension.
     */
    private ArrayList<Long> numIterations;

    /**
     * internal storage for the call.
     */
    private ComplexExpressionMapping definition;

    /**
     * Stores the processor executing this pattern.
     */
    private Processor executor;

    /**
     * Stores the number threads to be used.
     */
    private int numThreads;

    /**
     * Set, iff the parallel call executes a dynamic programming recursion. Defines the barrier executed after each time-step.
     */
    private Optional<BarrierMapping> dynamicProgrammingBarrier;

    /**
     * Set, iff the parallel call executes a dynamic programming recursion. Defines the data transfers necessary after each time-step.
     */
    private Set<DataMovementMapping> dynamicProgrammingdataTransfers;

    /**
     * stores the group of call nodes defining the complete pattern execution.
     */
    private ParallelGroup group;

    public ParallelCallMapping(Optional<MappingNode> parent, HashMap<String, Data> variableTable, CallNode aptNode, ArrayList<Long> startIndex, ArrayList<Long> numIterations, Processor executor, int numThreads, Optional<BarrierMapping> dynamicProgrammingBarrier, Set<DataMovementMapping> dynamicProgrammingdataTransfers) {
        super(parent, variableTable, aptNode);
        this.startIndex = startIndex;
        this.numIterations = numIterations;
        this.executor = executor;
        this.numThreads = numThreads;
        this.dynamicProgrammingBarrier = dynamicProgrammingBarrier;
        this.dynamicProgrammingdataTransfers = dynamicProgrammingdataTransfers;
    }

    public ArrayList<Long> getStartIndex() {
        return startIndex;
    }

    public ArrayList<Long> getNumIterations() {
        return numIterations;
    }

    public void setDefinition(ComplexExpressionMapping definition) {
        this.definition = definition;
    }

    public ComplexExpressionMapping getDefinition() {
        return definition;
    }

    public Processor getExecutor() {
        return executor;
    }

    public int getNumThreads() {
        return numThreads;
    }

    public ParallelGroup getGroup() {
        return group;
    }

    public void setGroup(ParallelGroup group) {
        this.group = group;
    }

    public Optional<BarrierMapping> getDynamicProgrammingBarrier() {
        return dynamicProgrammingBarrier;
    }

    public Set<DataMovementMapping> getDynamicProgrammingdataTransfers() {
        return dynamicProgrammingdataTransfers;
    }

    public ArrayList<OperationExpression> getArgumentExpressions() {

        ArrayList<OperationExpression> arguments = new ArrayList<>();

        int firstOperand = 1;
        int numOperands = 1;

        int firstOperator = 1;
        int numOperators = 0;

        OperationExpression call;

            IRLExpression exp =  definition.getExpression();
            if (exp instanceof AssignmentExpression) {
                call = ((AssignmentExpression) exp).getRhsExpression();
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

    @Override
    public HashSet<DataPlacement> getNecessaryData() {
        FunctionMapping functionMapping = AbstractMappingTree.getFunctionTable().get(getFunctionIdentifier());
        HashSet<DataPlacement> placements = new HashSet<>();
        for (int i = 0; i < functionMapping.getArgumentCount(); i++) {
            if (functionMapping.getArgumentValues().get(i) instanceof PrimitiveData) {
                ArrayList<EndPoint> endPoints = new ArrayList<>();
                EndPoint target = new EndPoint(executor.getParent(), 0,1, SupportFunction.getElementSet(group), false);
                endPoints.add(target);
                DataPlacement placement = new DataPlacement(endPoints, getArgumentExpressions().get(i).getOperands().get(0));
                placements.add(placement);
            } else if (functionMapping.getArgumentValues().get(i) instanceof ArrayData) {
                ArrayList<EndPoint> endPoints = new ArrayList<>();
                ArrayData value = (ArrayData) functionMapping.getArgumentValues().get(i);

                for (DataAccess access: value.getTrace().getDataAccesses() ) {
                    if (access instanceof MapDataAccess) {
                        if (((MapDataAccess) access).getScalingFactor() == 1) {
                            EndPoint endPoint = new EndPoint(executor.getParent(), ((MapDataAccess) access).getShiftOffset() + startIndex.get(0), numIterations.get(0), SupportFunction.getElementSet(group), false);
                            endPoints.add(endPoint);
                        } else {
                            for (long j = startIndex.get(0); j < startIndex.get(0) + numIterations.get(0); j++) {
                                EndPoint endPoint = new EndPoint(executor.getParent(), ((MapDataAccess) access).getScalingFactor() * j + ((MapDataAccess) access).getShiftOffset(), 1, SupportFunction.getElementSet(group), false);
                                endPoints.add(endPoint);
                            }
                        }
                    } else if (access instanceof DynamicProgrammingDataAccess){
                        EndPoint endPoint = new EndPoint(executor.getParent(), ((DynamicProgrammingDataAccess) access).getShiftOffsets().get(0) + startIndex.get(1), numIterations.get(1), SupportFunction.getElementSet(group), false);
                        endPoints.add(endPoint);
                    } else if (access instanceof StencilDataAccess) {
                        if (((StencilDataAccess) access).getScalingFactors().get(0) == 1) {
                            EndPoint endPoint = new EndPoint(executor.getParent(), ((StencilDataAccess) access).getShiftOffsets().get(0) + startIndex.get(0), numIterations.get(0), SupportFunction.getElementSet(group), false);
                            endPoints.add(endPoint);
                        } else {
                            for (long j = startIndex.get(0); j < startIndex.get(0) + numIterations.get(0); j++) {
                                EndPoint endPoint = new EndPoint(executor.getParent(), ((StencilDataAccess) access).getScalingFactors().get(0) * j + ((StencilDataAccess) access).getShiftOffsets().get(0), 1, SupportFunction.getElementSet(group), false);
                                endPoints.add(endPoint);
                            }
                        }
                    } else {
                        EndPoint target = new EndPoint(executor.getParent(), 0,value.getShape().get(0), SupportFunction.getElementSet(group), false);
                        endPoints.add(target);
                    }
                }

                DataPlacement placement = new DataPlacement(endPoints, getArgumentExpressions().get(i).getOperands().get(0));
                placements.add(placement);
            }
        }
        return placements;
    }

    @Override
    public HashSet<DataPlacement> getOutputData() {
        ParallelMapping functionMapping = (ParallelMapping) AbstractMappingTree.getFunctionTable().get(getFunctionIdentifier());
        HashSet<DataPlacement> placements = new HashSet<>();
        if (functionMapping.getReturnElement() instanceof PrimitiveData) {
            ArrayList<EndPoint> endPoints = new ArrayList<>();
            EndPoint target = new EndPoint(executor.getParent(), 0,1, SupportFunction.getElementSet(group), true);
            endPoints.add(target);
            DataPlacement placement = new DataPlacement(endPoints, ((AssignmentExpression) definition.getExpression()).getOutputElement());
            placements.add(placement);
        } else if (functionMapping.getReturnElement() instanceof ArrayData) {
            ArrayList<EndPoint> endPoints = new ArrayList<>();
            ArrayData value = (ArrayData) functionMapping.getReturnElement();

            for (DataAccess access: value.getTrace().getDataAccesses() ) {
                if (access instanceof MapDataAccess) {
                    if (((MapDataAccess) access).getScalingFactor() == 1) {
                        EndPoint endPoint = new EndPoint(executor.getParent(), ((MapDataAccess) access).getShiftOffset() + startIndex.get(0), numIterations.get(0), SupportFunction.getElementSet(group), true);
                        endPoints.add(endPoint);
                    } else {
                        for (long j = startIndex.get(0); j < startIndex.get(0) + numIterations.get(0); j++) {
                            EndPoint endPoint = new EndPoint(executor.getParent(), ((MapDataAccess) access).getScalingFactor() * j + ((MapDataAccess) access).getShiftOffset(), 1, SupportFunction.getElementSet(group), true);
                            endPoints.add(endPoint);
                        }
                    }
                } else if (access instanceof DynamicProgrammingDataAccess){
                    EndPoint endPoint = new EndPoint(executor.getParent(), ((DynamicProgrammingDataAccess) access).getShiftOffsets().get(0) + startIndex.get(1), numIterations.get(1), SupportFunction.getElementSet(group), true);
                    endPoints.add(endPoint);
                } else if (access instanceof StencilDataAccess) {
                    if (((StencilDataAccess) access).getScalingFactors().get(0) == 1) {
                        EndPoint endPoint = new EndPoint(executor.getParent(), ((StencilDataAccess) access).getShiftOffsets().get(0) + startIndex.get(0), numIterations.get(0), SupportFunction.getElementSet(group), true);
                        endPoints.add(endPoint);
                    } else {
                        for (long j = startIndex.get(0); j < startIndex.get(0) + numIterations.get(0); j++) {
                            EndPoint endPoint = new EndPoint(executor.getParent(), ((StencilDataAccess) access).getScalingFactors().get(0) * j + ((StencilDataAccess) access).getShiftOffsets().get(0), 1, SupportFunction.getElementSet(group), true);
                            endPoints.add(endPoint);
                        }
                    }
                } else {
                    EndPoint target = new EndPoint(executor.getParent(), 0,value.getShape().get(0), SupportFunction.getElementSet(group), true);
                    endPoints.add(target);
                }
            }
            DataPlacement placement = new DataPlacement(endPoints, ((AssignmentExpression) definition.getExpression()).getOutputElement());
            placements.add(placement);
        }
        return placements;
    }

    public void setDynamicProgrammingBarrier(Optional<BarrierMapping> dynamicProgrammingBarrier) {
        this.dynamicProgrammingBarrier = dynamicProgrammingBarrier;
    }

    public void setDynamicProgrammingdataTransfers(Set<DataMovementMapping> dynamicProgrammingdataTransfers) {
        this.dynamicProgrammingdataTransfers = dynamicProgrammingdataTransfers;
    }

    /**
     * Visitor accept function.
     */
    public void accept(AMTVisitor visitor) {
        visitor.handle(this);
    }
}
