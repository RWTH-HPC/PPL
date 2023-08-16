package de.parallelpatterndsl.patterndsl.Postprocessing;

import de.parallelpatterndsl.patterndsl.abstractPatternTree.AbstractPatternTree;
import de.parallelpatterndsl.patterndsl.abstractPatternTree.DataElements.ArrayData;
import de.parallelpatterndsl.patterndsl.abstractPatternTree.DataElements.Data;
import de.parallelpatterndsl.patterndsl.abstractPatternTree.DataElements.DataAccess.DataAccess;
import de.parallelpatterndsl.patterndsl.abstractPatternTree.DataElements.DataAccess.DynamicProgrammingDataAccess;
import de.parallelpatterndsl.patterndsl.abstractPatternTree.DataElements.DataAccess.MapDataAccess;
import de.parallelpatterndsl.patterndsl.abstractPatternTree.DataElements.DataAccess.StencilDataAccess;
import de.parallelpatterndsl.patterndsl.abstractPatternTree.DataElements.FunctionInlineData;
import de.parallelpatterndsl.patterndsl.abstractPatternTree.DataElements.FunctionReturnData;
import de.parallelpatterndsl.patterndsl.abstractPatternTree.Nodes.FunctionNode;
import de.parallelpatterndsl.patterndsl.abstractPatternTree.Nodes.Functions.DynamicProgrammingNode;
import de.parallelpatterndsl.patterndsl.abstractPatternTree.Nodes.Functions.MapNode;
import de.parallelpatterndsl.patterndsl.abstractPatternTree.Nodes.Functions.ReduceNode;
import de.parallelpatterndsl.patterndsl.abstractPatternTree.Nodes.Functions.StencilNode;
import de.parallelpatterndsl.patterndsl.abstractPatternTree.Nodes.Plain.AdditionalArguments.AdditionalArguments;
import de.parallelpatterndsl.patterndsl.abstractPatternTree.Nodes.Plain.AdditionalArguments.MetaList;
import de.parallelpatterndsl.patterndsl.abstractPatternTree.Nodes.Plain.AdditionalArguments.MetaValue;
import de.parallelpatterndsl.patterndsl.abstractPatternTree.Nodes.Plain.ComplexExpressionNode;
import de.parallelpatterndsl.patterndsl.abstractPatternTree.Nodes.Plain.ParallelCallNode;
import de.parallelpatterndsl.patterndsl.abstractPatternTree.Nodes.Plain.SimpleExpressionBlockNode;
import de.parallelpatterndsl.patterndsl.abstractPatternTree.Visitor.ExtendedShapeAPTVisitor;
import de.parallelpatterndsl.patterndsl.expressions.AssignmentExpression;
import de.parallelpatterndsl.patterndsl.expressions.Operator;
import de.se_rwth.commons.logging.Log;

import java.util.ArrayList;

public class APTAdditionalArgumentGeneration {


    public void generate() {
        AdditionalArgumentGenerator gen = new AdditionalArgumentGenerator();

        AbstractPatternTree.getFunctionTable().get("main").accept(gen);
        Log.info("additional argument generation finished!", "");
    }

    /**
     * Support class used to generate the computable additional arguments for all parallel calls.
     */
    private class AdditionalArgumentGenerator implements ExtendedShapeAPTVisitor {

        @Override
        public void visit(ParallelCallNode node) {
            String name = node.getFunctionIdentifier();

            FunctionNode function = AbstractPatternTree.getFunctionTable().get(name);

            ArrayList<AdditionalArguments> additionalArguments = new ArrayList<>();

            if (function instanceof MapNode) {
                MapNode mapNode = (MapNode) function;

                MetaValue<Long> start = new MetaValue<>(getMapStart(mapNode,node));

                MetaValue<Long> width = new MetaValue<>(getMapWidth(mapNode,node,start.getValue()));

                additionalArguments.add(width);

                additionalArguments.add(start);
            } else if (function instanceof ReduceNode) {
                ReduceNode reduceNode = (ReduceNode) function;

                long start = getReductionStart(reduceNode,node);

                long width = getReductionWidth(reduceNode,node,start);

                long arity = getReductionArity(reduceNode);

                long depth = getReductionDepth(width,arity);

                ArrayList<Long> list = new ArrayList<>();

                list.add(width);
                list.add(arity);
                list.add(depth);
                list.add(start);

                MetaList<Long> meta = new MetaList<>(list);

                additionalArguments.add(meta);
            } else if (function instanceof DynamicProgrammingNode) {
                DynamicProgrammingNode dbNode = (DynamicProgrammingNode) function;

                // the time steps are the first additional argument for dynamic programming
                MetaValue<Long> timesteps = (MetaValue<Long>) node.getAdditionalArguments().get(0);

                node.getAdditionalArguments().remove(0);

                additionalArguments.add(timesteps);

                MetaList<Long> starts = new MetaList<>(getDynamicProgrammingStart(dbNode,node));

                MetaValue<Long> width = new MetaValue<>(getDynamicProgrammingWidth(dbNode,node,starts.getValues()));

                additionalArguments.add(width);

                additionalArguments.add(starts);

            } else if (function instanceof StencilNode) {
                StencilNode stencilNode = (StencilNode) function;

                MetaList<Long> starts = new MetaList<>(getStencilStarts(stencilNode,node));

                MetaList<Long> widths = new MetaList<>(getStencilWidths(stencilNode,node,starts.getValues()));

                additionalArguments.add(widths);
                additionalArguments.add(starts);

            }

            additionalArguments.addAll(node.getAdditionalArguments());

            node.setAdditionalArguments(additionalArguments);

        }

        /**
         * Visitor support functions.
         */
        private ExtendedShapeAPTVisitor realThis = this;

        @Override
        public ExtendedShapeAPTVisitor getRealThis() {
            return realThis;
        }

        public void setRealThis(ExtendedShapeAPTVisitor realThis) {
            this.realThis = realThis;
        }
    }

    /**
     * Returns the width (number of instances/iterations) of a stencil call.
     * @param function
     * @param call
     * @param initialOffsets
     * @return
     */
    private ArrayList<Long> getStencilWidths(StencilNode function, ParallelCallNode call, ArrayList<Long> initialOffsets) {
        ArrayList<Long> widths = new ArrayList<>();
        int maxDimensions = 1;
        for (int dim = 0; dim < function.getDimension(); dim++) {
            long N = Long.MAX_VALUE;
            String currentRuleBase = "INDEX" + dim;
            if (function.getVariableTable().get(currentRuleBase).getTrace().getDataAccesses().isEmpty()) {
                break;
            }
            for (int i = 0; i < call.getParameterCount() + 1; ++i) {
                Data argumentData;
                if (i == call.getParameterCount()) {
                    argumentData = function.getReturnElement();
                } else {
                    argumentData = function.getArgumentValues().get(i);
                }
                if (!(argumentData instanceof ArrayData)) {
                    continue;
                }
                ArrayList<DataAccess> accesses = argumentData.getTrace().getDataAccesses();
                for (DataAccess access : accesses) {
                    // Replace by correct access class.
                    if (access instanceof StencilDataAccess) {
                        StencilDataAccess stencilAccess = ((StencilDataAccess) access);
                        maxDimensions = Math.max(maxDimensions, stencilAccess.getRuleBaseIndex().size());
                        for (int j = 0; j < stencilAccess.getRuleBaseIndex().size(); j++) {
                            if (stencilAccess.getRuleBaseIndex().get(j).equals(currentRuleBase)) {
                                N = Long.min((long) Math.ceil((((ArrayData) argumentData).getShape().get(j) - initialOffsets.get(dim) - Math.max(0, stencilAccess.getShiftOffsets().get(j)))/ (double) stencilAccess.getScalingFactors().get(j)), N);
                            }
                        }
                    }
                }
            }
            widths.add(N);
        }
        function.setDimension(maxDimensions);

        return widths;
    }

    /**
     * Returns the minimal initial offset to avoid negative indices for a stencil call.
     * @param function
     * @param call
     * @return
     */
    private ArrayList<Long> getStencilStarts(StencilNode function, ParallelCallNode call) {
        ArrayList<Long> starts = new ArrayList<>();
        for (int dim = 0; dim < function.getDimension(); dim++) {
            boolean hasSpecStart = false;
            long N = Long.MIN_VALUE;
            String currentRuleBase = "INDEX" + dim;
            if (function.getVariableTable().get(currentRuleBase).getTrace().getDataAccesses().isEmpty()) {
                function.setDimension(dim);
                break;
            }
            for (int i = 0; i < call.getParameterCount() + 1; ++i) {
                Data argumentData;
                if (i == call.getParameterCount()) {
                    argumentData = function.getReturnElement();
                } else {
                    argumentData = function.getArgumentValues().get(i);
                }
                if (!(argumentData instanceof ArrayData)) {
                    continue;
                }
                ArrayList<DataAccess> accesses = argumentData.getTrace().getDataAccesses();
                for (DataAccess access : accesses) {
                    // Replace by correct access class.
                    if (access instanceof StencilDataAccess) {
                        StencilDataAccess stencilAccess = ((StencilDataAccess) access);
                        for (int j = 0; j < stencilAccess.getRuleBaseIndex().size(); j++) {
                            if (stencilAccess.getRuleBaseIndex().get(j).equals(currentRuleBase) && stencilAccess.getShiftOffsets().get(j) < 0) {
                                hasSpecStart = true;
                                long NOld = N;
                                N = Long.max((long) Math.floor((Math.abs(stencilAccess.getShiftOffsets().get(j))) / (double) stencilAccess.getScalingFactors().get(j)), N);
                                if ((Math.abs(stencilAccess.getShiftOffsets().get(j))) % stencilAccess.getScalingFactors().get(j) != 0 && (int) Math.floor((Math.abs(stencilAccess.getShiftOffsets().get(j))) / (double) stencilAccess.getScalingFactors().get(j)) >= NOld) {
                                    N++;
                                }
                            }
                        }
                    }
                }
            }
            if (!hasSpecStart) {
                N=0;
            }
            starts.add(N);
        }
        return starts;
    }

    /**
     *
     * Returns the width (number of instances/iterations) of a dynamic programming call.
     * This function may throw an error, if a data element is accessed based on time and the array is to small for a given number of time steps.
     * @param function
     * @param call
     * @param initialOffsets
     * @return
     */

    private long getDynamicProgrammingWidth(DynamicProgrammingNode function, ParallelCallNode call, ArrayList<Long> initialOffsets) {
        long N = Long.MAX_VALUE;
        boolean hasInternal = false;

        for (int i = 0; i < call.getParameterCount() + 1; ++i) {
            Data argumentData;
            if (i == call.getParameterCount()) {
                argumentData = function.getReturnElement();
            } else {
                argumentData = function.getArgumentValues().get(i);
            }
            if (!(argumentData instanceof ArrayData)) {
                continue;
            }

            ArrayList<DataAccess> accesses = argumentData.getTrace().getDataAccesses();
            for (DataAccess access : accesses) {
                // Replace by correct access class.
                if (access instanceof DynamicProgrammingDataAccess) {
                    DynamicProgrammingDataAccess dpAccess = ((DynamicProgrammingDataAccess) access);

                    if (dpAccess.getRuleBaseIndex().get(0).equals("INDEX0")) {
                        // get the number of time steps, since INDEX0 iterates over the time steps.
                        long diff = ((MetaValue<Integer>) call.getAdditionalArguments().get(0)).getValue() - (((ArrayData) argumentData).getShape().get(0) - dpAccess.getShiftOffsets().get(0) - initialOffsets.get(0));
                        if (diff > 0){
                            Log.error("Size of array " + argumentData.getIdentifier() + " to small for this number of time steps! Try " + diff + " more elements!");
                            throw new RuntimeException("Critical error!");
                        }
                    } else if (dpAccess.getRuleBaseIndex().get(0).equals("INDEX1")) {
                        N = Long.min((long) ((ArrayData) argumentData).getShape().get(0) - Math.max(0, dpAccess.getShiftOffsets().get(0)) - initialOffsets.get(1), N);
                        hasInternal = true;
                    }
                }
            }
        }
        if (hasInternal) {
            return N;
        }
        return 1;
    }


    /**
     * Returns the minimal initial offset to avoid negative indices for a dynamic programming call.
     * @param function
     * @param call
     * @return
     */
    private ArrayList<Long> getDynamicProgrammingStart(DynamicProgrammingNode function, ParallelCallNode call) {

        ArrayList<Long> starts = new ArrayList<>();
        for (int dim = 0; dim < 2; dim++) {
            long N = Long.MIN_VALUE;
            boolean hasSpecStart = false;
            String currentRuleBase = "INDEX" + dim;
            for (int i = 0; i < call.getParameterCount() + 1; ++i) {
                Data argumentData;
                if (i == call.getParameterCount()) {
                    argumentData = function.getReturnElement();
                } else {
                    argumentData = function.getArgumentValues().get(i);
                }
                if (!(argumentData instanceof ArrayData)) {
                    continue;
                }

                ArrayList<DataAccess> accesses = argumentData.getTrace().getDataAccesses();
                for (DataAccess access : accesses) {
                    // Replace by correct access class.
                    if (access instanceof DynamicProgrammingDataAccess) {
                        DynamicProgrammingDataAccess dpAccess = ((DynamicProgrammingDataAccess) access);
                        if (dpAccess.getShiftOffsets().get(0) < 0 && dpAccess.getRuleBaseIndex().get(0).equals(currentRuleBase)) {
                            hasSpecStart = true;
                            N = Long.max(-dpAccess.getShiftOffsets().get(0), N);
                        }
                    }
                }
            }
            if (!hasSpecStart) {
                N=0;
            }
            starts.add(N);
        }
        return starts;
    }

    /**
     * Returns the width (number of instances/iterations) of a Map call.
     * @param function
     * @param call
     * @param initialOffset
     * @return
     */
    private long getMapWidth(MapNode function, ParallelCallNode call, long initialOffset) {
        long N = Long.MAX_VALUE;
        for (int i = 0; i < call.getParameterCount() + 1; ++i) {
            Data argumentData;
            if (i == call.getParameterCount()) {
                argumentData = function.getReturnElement();
            } else {
                argumentData = function.getArgumentValues().get(i);
            }
            if (!(argumentData instanceof ArrayData)) {
                continue;
            }

            ArrayList<DataAccess> accesses = argumentData.getTrace().getDataAccesses();
            for (DataAccess access : accesses) {
                // Replace by correct access class.
                if (access instanceof MapDataAccess) {
                    MapDataAccess mapAccess = ((MapDataAccess) access);
                    if (mapAccess.getShiftOffset() >= 0) {
                        N = Long.min((int) Math.ceil((((ArrayData) argumentData).getShape().get(0) - Math.max(mapAccess.getShiftOffset(), 0) - initialOffset) / (double) mapAccess.getScalingFactor()), N);
                    }
                }
            }
        }
        return N;
    }

    /**
     * Returns the minimal initial offset to avoid negative indices for a map call.
     * @param function
     * @param call
     * @return
     */
    private long getMapStart(MapNode function, ParallelCallNode call) {
        long N = Long.MIN_VALUE;
        boolean hasSpecStart = false;
        for (int i = 0; i < call.getParameterCount() + 1; ++i) {
            Data argumentData;
            if (i == call.getParameterCount()) {
                argumentData = function.getReturnElement();
            } else {
                argumentData = function.getArgumentValues().get(i);
            }
            if (!(argumentData instanceof ArrayData)) {
                continue;
            }

            ArrayList<DataAccess> accesses = argumentData.getTrace().getDataAccesses();
            for (DataAccess access : accesses) {
                // Replace by correct access class.
                if (access instanceof MapDataAccess) {
                    MapDataAccess mapAccess = ((MapDataAccess) access);
                    if (mapAccess.getShiftOffset() < 0) {
                        long NOld = N;
                        hasSpecStart = true;
                        N = Long.max((int) Math.floor(( Math.abs(mapAccess.getShiftOffset())) / (double) mapAccess.getScalingFactor()), N);
                        if (((Math.abs(mapAccess.getShiftOffset())) % mapAccess.getScalingFactor()) != 0 && (int) Math.floor(( Math.abs(mapAccess.getShiftOffset())) / (double) mapAccess.getScalingFactor()) >= NOld) {
                            N++;
                        }
                    }
                }
            }
        }

        if (!hasSpecStart) {
            N=0;
        }
        return N ;
    }

    /**
     * Returns the width (number of instances/iterations) of a reduction call.
     * @param function
     * @param call
     * @param initialOffset
     * @return
     */
    private long getReductionWidth(ReduceNode function, ParallelCallNode call, long initialOffset) {
        long N = Long.MAX_VALUE;
        for (int i = 0; i < call.getParameterCount(); ++i) {
            Data argumentData = function.getArgumentValues().get(i);
            if (!(argumentData instanceof ArrayData)) {
                continue;
            }

            ArrayList<DataAccess> accesses = argumentData.getTrace().getDataAccesses();
            for (DataAccess access : accesses) {
                // Replace by correct access class.
                if (access instanceof MapDataAccess) {
                    MapDataAccess mapAccess = ((MapDataAccess) access);
                    if (mapAccess.getShiftOffset() >= 0) {
                        N = Long.min((int) Math.ceil((((ArrayData) argumentData).getShape().get(0) - initialOffset - Math.max(0, mapAccess.getShiftOffset())) / (double) mapAccess.getScalingFactor()), N);
                    }
                }
            }
        }
        return N;
    }

    /**
     * Returns the minimal initial offset to avoid negative indices for a reduction call.
     * @param function
     * @param call
     * @return
     */
    private long getReductionStart(ReduceNode function, ParallelCallNode call) {
        long N = Long.MIN_VALUE;
        boolean hasSpecStart = false;
        for (int i = 0; i < call.getParameterCount(); ++i) {
            //Data inputData = call.getInputElements().get(i);
            Data argumentData = function.getArgumentValues().get(i);
            if (!(argumentData instanceof ArrayData)) {
                continue;
            }

            ArrayList<DataAccess> accesses = argumentData.getTrace().getDataAccesses();
            for (DataAccess access : accesses) {
                // Replace by correct access class.
                if (access instanceof MapDataAccess) {
                    MapDataAccess mapAccess = ((MapDataAccess) access);
                    if (mapAccess.getShiftOffset() < 0) {
                        long NOld = N;
                        hasSpecStart = true;
                        N = Long.max((int) Math.floor(( Math.abs(mapAccess.getShiftOffset())) / (double) mapAccess.getScalingFactor()), N);
                        if (((Math.abs(mapAccess.getShiftOffset())) % mapAccess.getScalingFactor()) != 0 && (int) Math.floor(( Math.abs(mapAccess.getShiftOffset())) / (double) mapAccess.getScalingFactor()) >= NOld) {
                            N++;
                        }
                    }
                }
            }
        }

        if (!hasSpecStart) {
            N=0;
        }
        return N ;
    }

    /**
     * Calculates the depth of the reduction (The number of necessary synchronization steps).
     * @param width
     * @param arity
     * @return
     */
    private long getReductionDepth(long width, long arity) {
        long depth = (long) (Math.log10(width) / Math.log10(arity));
        return depth;
    }


    /**
     * Computes the arity of the reduction step from a given reduce node.
     * The arity is defined as the number of read data elements within the reduction step. e.g. "res += in1 * in2" has an arity of two, because in1 and in2 are both accessed once.
     * @param node
     * @return
     */
    private long getReductionArity(ReduceNode node) {
        if (node.getChildren().get(node.getChildren().size() - 1) instanceof SimpleExpressionBlockNode) {
            SimpleExpressionBlockNode expNode = (SimpleExpressionBlockNode) node.getChildren().get(node.getChildren().size() - 1);
            AssignmentExpression exp = (AssignmentExpression) expNode.getExpressionList().get(expNode.getExpressionList().size() - 1);
            return Operator.arity(exp.getOperator());
        } else if (node.getChildren().get(node.getChildren().size() - 1) instanceof ComplexExpressionNode) {
            AssignmentExpression exp = (AssignmentExpression) ((ComplexExpressionNode) node.getChildren().get(node.getChildren().size() - 1)).getExpression();
            if (exp.getRhsExpression().getOperands().get(0) instanceof FunctionInlineData) {
                // return for min and max
                return 2;
            }
            return exp.getRhsExpression().getOperands().size();
        } else {
            Log.error("Reduction not sufficiently defined!  " + node.getIdentifier());
            throw new RuntimeException("Critical error!");
        }
    }
}
