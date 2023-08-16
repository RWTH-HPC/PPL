package de.parallelpatterndsl.patterndsl.OPT2AMT;

import de.parallelpatterndsl.patterndsl.FlatAPT;
import de.parallelpatterndsl.patterndsl.MappingTree.AbstractMappingTree;
import de.parallelpatterndsl.patterndsl.MappingTree.Nodes.Function.*;
import de.parallelpatterndsl.patterndsl.MappingTree.Nodes.FunctionMapping;
import de.parallelpatterndsl.patterndsl.MappingTree.Nodes.MappingNode;
import de.parallelpatterndsl.patterndsl.MappingTree.Nodes.ParallelCalls.*;
import de.parallelpatterndsl.patterndsl.MappingTree.Nodes.Plain.*;
import de.parallelpatterndsl.patterndsl.abstractPatternTree.AbstractPatternTree;
import de.parallelpatterndsl.patterndsl.abstractPatternTree.DataElements.TempData;
import de.parallelpatterndsl.patterndsl.abstractPatternTree.Nodes.FunctionNode;
import de.parallelpatterndsl.patterndsl.abstractPatternTree.Nodes.Functions.*;
import de.parallelpatterndsl.patterndsl.abstractPatternTree.Nodes.PatternNode;
import de.parallelpatterndsl.patterndsl.abstractPatternTree.Nodes.Plain.*;
import de.parallelpatterndsl.patterndsl.abstractPatternTree.Nodes.Plain.AdditionalArguments.MetaList;
import de.parallelpatterndsl.patterndsl.abstractPatternTree.Nodes.Plain.AdditionalArguments.MetaValue;
import de.parallelpatterndsl.patterndsl.dataSplits.TempDataSplit;
import de.parallelpatterndsl.patterndsl.hardwareDescription.Device;
import de.parallelpatterndsl.patterndsl.mapping.Mapping;
import de.parallelpatterndsl.patterndsl.mapping.StepMapping;
import de.parallelpatterndsl.patterndsl.patternSplits.*;
import de.parallelpatterndsl.patterndsl.teams.Team;
import de.se_rwth.commons.logging.Log;

import java.util.*;

public class OPT2AMTGenerator {

    private AbstractPatternTree APT;

    private FlatAPT flatRoot;

    private Mapping parallelMapping;

    private HashMap<String, FunctionMapping> functionTable;

    private int currentParallelStep;

    public OPT2AMTGenerator(AbstractPatternTree APT, FlatAPT flatRoot, Mapping parallelMapping) {
        this.APT = APT;
        this.flatRoot = flatRoot;
        this.parallelMapping = parallelMapping;
        currentParallelStep = 1;
        functionTable = new HashMap<>();
    }

    public AbstractMappingTree generate() {
        for (FunctionNode node : AbstractPatternTree.getFunctionTable().values() ) {
            if (node.getIdentifier().equals("main")) {
                functionTable.put(node.getIdentifier(), generateAMT(flatRoot, Optional.of(parallelMapping), true, node));
            } else {
                if (node.isAvailableAfterInlining()) {
                    functionTable.put(node.getIdentifier(), generateAMT(flatRoot, Optional.empty(), false, node));
                }
            }
        }
        AbstractMappingTree.setFunctionTable(functionTable);
        AbstractMappingTree AMT = new AbstractMappingTree((MainMapping) AbstractMappingTree.getFunctionTable().get("main"), APT.getGlobalVariableTable(), APT.getGlobalAssignments());
        return AMT;
    }


    public FunctionMapping generateAMT(FlatAPT flatAPT, Optional<Mapping> mappingOPT, boolean isMain, FunctionNode node) {
        Mapping mapping;

        FunctionMapping functionMapping;
        if (node instanceof MapNode) {
            functionMapping = new MapMapping((MapNode) node);
        } else if (node instanceof DynamicProgrammingNode) {
            functionMapping = new DynamicProgrammingMapping((DynamicProgrammingNode) node);
        } else if (node instanceof MainNode) {
            functionMapping = new MainMapping((MainNode) node);
        } else if (node instanceof RecursionNode) {
            functionMapping = new RecursionMapping((RecursionNode) node);
        } else if (node instanceof ReduceNode) {
            functionMapping = new ReduceMapping((ReduceNode) node);
        } else if (node instanceof SerialNode) {
            functionMapping = new SerialMapping((SerialNode) node);
        } else if (node instanceof StencilNode) {
            functionMapping = new StencilMapping((StencilNode) node);
        } else {
            Log.error("Function type not defined! " + node.getIdentifier());
            throw new RuntimeException("Critical error!");
        }


        if (isMain) {
            if (mappingOPT.isPresent()) {
                mapping = mappingOPT.get();


                functionMapping = new MainMapping(node);
                ArrayList<MappingNode> children = new ArrayList<>();
                for (int i = 0; i < flatAPT.size(); i++) {

                    //Tests, if the current parallel step is within the same global step, as the current serial step.
                    if (mapping.currentStep() > currentParallelStep && isInSameStep(flatAPT.getSplits(i), mapping.assignmentOf(currentParallelStep))) {
                        children.addAll(generateParallelStep(mapping.assignmentOf(currentParallelStep), functionMapping));
                        currentParallelStep++;
                    }


                    children.addAll(generateSerialStep(flatAPT.getSplits(i), functionMapping));

                }

                functionMapping.setChildren(children);
            } else {
                Log.error("No mapping for main generated!");
                throw new RuntimeException("Critical error!");
            }
        } else {

            ArrayList<MappingNode> children = new ArrayList<>();
            FunctionNode aptFunctionNode = AbstractPatternTree.getFunctionTable().get(functionMapping.getIdentifier());
                for (PatternNode aptNode : aptFunctionNode.getChildren() ) {
                    children.add(generateMappingNode(aptNode, functionMapping, Optional.empty()));
                }
            functionMapping.setChildren(children);
        }
        return functionMapping;
    }

    private boolean isInSameStep(Set<PatternSplit> serialStep, StepMapping parallelStep) {
        for (PatternSplit split: parallelStep.splits() ) {
            if (split instanceof FusedPatternSplit) {
                if (serialStep.contains(((FusedPatternSplit) split).getJobs().getFirst())) {
                    return true;
                }
            } else {
                if (serialStep.contains(split)) {
                    return true;
                }
            }
        }
        return false;
    }

    /**
     * Generates the Serial mapping nodes for a specific step.
     * @param step
     * @param parent
     * @return
     */
    private ArrayList<MappingNode> generateSerialStep(Set<PatternSplit> step, FunctionMapping parent) {
        ArrayList<MappingNode> serialNodes = new ArrayList<>();
        for (Object obj : step.stream().filter(x -> (x instanceof SerialPatternSplit)).toArray() ) {
            PatternNode node = ((SerialPatternSplit) obj).getNode();
            serialNodes.add(generateMappingNode(node, parent, Optional.empty()));
        }
        return serialNodes;
    }


    /**
     * This function generates the list of parallel mapping nodes for a given parallel step.
     * @param step
     * @param parent
     * @return
     */
    private ArrayList<MappingNode> generateParallelStep(StepMapping step, FunctionMapping parent) {
        ArrayList<MappingNode> parallelNodes = new ArrayList<>();
        for (ArrayList<PatternSplit> splits: generateConcatenatedSplits(fuseSplits(step))) {
            PatternSplit represents = splits.get(0);
            if (represents instanceof IOPatternSplit) {
                continue;
            }

            //Generate Fused Pattern mappings under the assumption that all fused splits are the same with different ranges.
            if (represents instanceof FusedPatternSplit) {
                Team execute = step.get(represents);
                // Generate GPU execution
                if (execute.getDevice().getType().equals("GPU")) {
                    FusedParallelCallMapping mapping = new FusedParallelCallMapping(Optional.of(parent), represents.getNode().getVariableTable());
                    ArrayList<MappingNode> children = new ArrayList<>();
                    for (int i = 0; i < ((FusedPatternSplit) represents).getJobs().size(); i++) {
                        // the list horizontalFusing is assumed to be concatenated.
                        ArrayList<ParallelPatternSplit> horizontalFusing = new ArrayList<>();
                        for (PatternSplit split: splits) {
                            if (split instanceof FusedPatternSplit) {
                                horizontalFusing.add(((FusedPatternSplit) split).getJobs().get(i));
                            }
                        }
                        // generate reductions under the assumption, that on the GPU only a single node is generated
                        if (AbstractPatternTree.getFunctionTable().get(horizontalFusing.get(0).getNode().getFunctionIdentifier()) instanceof ReduceNode) {
                            children.add(generateReductionMapping(step, horizontalFusing, mapping, Optional.of(execute)).get(0));
                        } else {
                            children.add(generateGPUMapping(step, horizontalFusing, mapping, Optional.of(execute)));
                        }
                    }
                    mapping.setChildren(children);
                    parallelNodes.add(mapping);
                } else {
                    for (PatternSplit split: splits ) {
                        FusedParallelCallMapping mapping = new FusedParallelCallMapping(Optional.of(parent), represents.getNode().getVariableTable());
                        ArrayList<MappingNode> children = new ArrayList<>();
                        if (split instanceof FusedPatternSplit) {

                            // generate the individual steps under the assumption, that the generator function only generate a single mapping for a single input.
                            for (ParallelPatternSplit job: ((FusedPatternSplit) split).getJobs() ) {
                                ArrayList<ParallelPatternSplit> wrapper = new ArrayList<>();
                                wrapper.add(job);

                                // generate reductions under the assumption, that on the GPU only a single node is generated
                                if (AbstractPatternTree.getFunctionTable().get(job.getNode().getFunctionIdentifier()) instanceof ReduceNode) {
                                    children.add(generateReductionMapping(step, wrapper, mapping, Optional.of(execute)).get(0));
                                } else {
                                    children.add(generatePlainParallelCallMapping(step, wrapper, mapping, Optional.of(execute)).get(0));
                                }
                            }
                        }
                        mapping.setChildren(children);
                        parallelNodes.add(mapping);
                    }
                }
                // generate non-fused mappings
            } else {
                ArrayList<ParallelPatternSplit> parallelSplits = new ArrayList<>();
                for (PatternSplit split: splits ) {
                    if (split instanceof ParallelPatternSplit) {
                        parallelSplits.add((ParallelPatternSplit) split);
                    }
                }
                // Generate GPU execution
                if (step.get(represents).getDevice().getType().equals("GPU")) {
                        if (AbstractPatternTree.getFunctionTable().get(((ParallelPatternSplit) represents).getNode().getFunctionIdentifier()) instanceof ReduceNode) {
                            parallelNodes.addAll(generateReductionMapping(step, parallelSplits, parent, Optional.empty()));
                        } else {
                            parallelNodes.add(generateGPUMapping(step, parallelSplits, parent, Optional.empty()));
                        }
                } else {
                        if (AbstractPatternTree.getFunctionTable().get(((ParallelPatternSplit) represents).getNode().getFunctionIdentifier()) instanceof ReduceNode) {
                            parallelNodes.addAll(generateReductionMapping(step, parallelSplits, parent, Optional.empty()));
                        } else {
                            parallelNodes.addAll(generatePlainParallelCallMapping(step, parallelSplits, parent, Optional.empty()));
                        }
                }
            }
        }
        return parallelNodes;
    }


    /**
     * Generates a set of simple parallel call mappings.
     * @param step
     * @param split
     * @param parent
     * @return
     */
    private ArrayList<ParallelCallMapping> generatePlainParallelCallMapping(StepMapping step, ArrayList<ParallelPatternSplit> split, MappingNode parent, Optional<Team> executorOPT) {
        ArrayList<Long> iterations = new ArrayList<>();
        int length = split.get(0).getStartIndices().length;
        for (int i = 0; i < length; i++) {
            long numIterations = split.get(split.size() - 1).getStartIndices()[i] + split.get(split.size() - 1).getLengths()[i] - split.get(0).getStartIndices()[i];
            iterations.add(numIterations);
        }
        ArrayList<Long> starts = new ArrayList<>();
        for (long start: split.get(0).getStartIndices() ) {
            starts.add(start);
        }

        ArrayList<ParallelCallMapping> res = new ArrayList<>();

        //for (ParallelPatternSplit call: split) {
        ParallelPatternSplit call = split.get(0);
            ParallelCallMapping result;
            Team execute = executorOPT.orElseGet(() -> step.get(call));
            result = new ParallelCallMapping(Optional.of(parent), parent.getVariableTable(), call.getNode(), starts, iterations, execute.getProcessor(), execute.getCores(), Optional.empty(), new HashSet<>());

            result.setDefinition(generateComplexExpressionMapping((ComplexExpressionNode) call.getNode().getChildren().get(0), result));
            result.setChildren(new ArrayList<>());
            res.add(result);
        //}

        return res;
    }


    /**
     * Generates the set of Reduction mapping nodes based on the given parallel pattern splits.
     * @param step
     * @param split
     * @param parent
     * @return
     */
    private ArrayList<ReductionCallMapping> generateReductionMapping (StepMapping step, ArrayList<ParallelPatternSplit> split, MappingNode parent, Optional<Team> executorOPT) {
        ArrayList<Long> iterations = new ArrayList<>();
        ArrayList<Long> starts = new ArrayList<>();
        Team execute = executorOPT.orElseGet(() -> step.get(split.get(0)));
        ArrayList<ReductionCallMapping> res = new ArrayList<>();

        if (execute.getDevice().getType().equals("GPU")) {
            int length = split.get(0).getLengths().length;
            for (int i = 0; i < length; i++) {
                long numIterations = split.get(split.size() - 1).getStartIndices()[i] + split.get(split.size() - 1).getLengths()[i] - split.get(0).getStartIndices()[i];
                iterations.add(numIterations);
            }
            for (long start: split.get(0).getStartIndices() ) {
                starts.add(start);
            }
            ReductionCallMapping result;
            HashSet<TempData> inputs = new HashSet<>();
            HashSet<TempData> outputs = new HashSet<>();
            for (ParallelPatternSplit call: split) {
                for (Object obj: call.getOutputDataSplits().stream().filter(x -> x instanceof TempDataSplit).toArray()) {
                    TempDataSplit temp = (TempDataSplit) obj;
                    TempData tempData = new TempData(call.getNode().getCallExpression().getTypeName(), temp.getIdentifier());
                    parent.getVariableTable().put("tempData_" + temp.getIdentifier(), tempData);
                    outputs.add(tempData);
                }
                for (Object obj: call.getInputDataSplits().stream().filter(x -> x instanceof TempDataSplit).toArray()) {
                    TempDataSplit temp = (TempDataSplit) obj;
                    inputs.add((TempData) parent.getVariableTable().get("tempData_" + temp.getIdentifier()));
                }
            }

            result = new ReductionCallMapping(Optional.of(parent), parent.getVariableTable(), split.get(0).getNode(), starts, iterations, execute.getProcessor(), execute.getCores(), !inputs.isEmpty(), inputs, outputs, split.size(), true);
            result.setDefinition(generateComplexExpressionMapping((ComplexExpressionNode) split.get(0).getNode().getChildren().get(0), result));
            result.setChildren(new ArrayList<>());
            res.add(result);
        } else {
            for (int j = 0; j < split.size(); j++) {
                ParallelPatternSplit call = split.get(j);
                iterations = new ArrayList<>();
                starts = new ArrayList<>();
                int length = split.get(j).getLengths().length;
                for (long numIterations: call.getLengths()) {
                    iterations.add(numIterations);
                }
                for (long start: call.getStartIndices() ) {
                    starts.add(start);
                }
                ReductionCallMapping result;
                HashSet<TempData> inputs = new HashSet<>();
                HashSet<TempData> outputs = new HashSet<>();
                for (Object obj: call.getOutputDataSplits().stream().filter(x -> x instanceof TempDataSplit).toArray()) {
                    TempDataSplit temp = (TempDataSplit) obj;
                    TempData tempData = new TempData(call.getNode().getCallExpression().getTypeName(), temp.getIdentifier());
                    parent.getVariableTable().put("tempData_" + temp.getIdentifier(), tempData);
                    outputs.add(tempData);
                }
                for (Object obj: call.getInputDataSplits().stream().filter(x -> x instanceof TempDataSplit).toArray()) {
                    TempDataSplit temp = (TempDataSplit) obj;
                    inputs.add((TempData) parent.getVariableTable().get("tempData_" + temp.getIdentifier()));
                }
                execute = executorOPT.orElseGet(() -> step.get(call));
                result = new ReductionCallMapping(Optional.of(parent), parent.getVariableTable(), call.getNode(), starts, iterations, execute.getProcessor(), execute.getCores(), !inputs.isEmpty(), inputs, outputs, split.size(), false);
                result.setDefinition(generateComplexExpressionMapping((ComplexExpressionNode) call.getNode().getChildren().get(0), result));
                result.setChildren(new ArrayList<>());
                res.add(result);
            }

        }

        return res;
    }

    /**
     * Generates a GPU mapping based on a sorted list of pattern splits. The splits must be concatenated to generate correct code.
     * @param step
     * @param split
     * @param parent
     * @return
     */
    private GPUParallelCallMapping generateGPUMapping(StepMapping step, ArrayList<ParallelPatternSplit> split, MappingNode parent, Optional<Team> executorOPT) {
        ArrayList<Long> iterations = new ArrayList<>();
        int length = split.get(0).getStartIndices().length;
        for (int i = 0; i < length; i++) {
            long numIterations = split.get(split.size() - 1).getStartIndices()[i] + split.get(split.size() - 1).getLengths()[i] - split.get(0).getStartIndices()[i];
            iterations.add(numIterations);
        }
        ArrayList<Long> starts = new ArrayList<>();
        for (long start: split.get(0).getStartIndices() ) {
            starts.add(start);
        }

        Team execute = executorOPT.orElseGet(() -> step.get(split.get(0)));

        GPUParallelCallMapping result;

        result = new GPUParallelCallMapping(Optional.of(parent), split.get(0).getNode().getVariableTable(), split.get(0).getNode(), starts, iterations, execute.getProcessor(), execute.getCores(), Optional.empty(), new HashSet<>(), split.size());


        result.setChildren(new ArrayList<>());

        result.setDefinition(generateComplexExpressionMapping((ComplexExpressionNode) split.get(0).getNode().getChildren().get(0), result));

        return result;
    }


    /**
     * Splits all sorted lists of pattern splits into sub-lists such that the iterations over the indices does not have gaps.
     * @param sortedSplits
     * @return
     */
    private Set<ArrayList<PatternSplit>> generateConcatenatedSplits(Set<ArrayList<PatternSplit>> sortedSplits) {
        Set<ArrayList<PatternSplit>> result = new HashSet<>();
        for (ArrayList<PatternSplit> splits: sortedSplits ) {
            result.addAll(generateConcatenatedSplit(splits));
        }
        return result;
    }


    /**
     * Splits the sorted list of pattern splits in a set of sub-lists, which are concatenated.
     * @param split
     * @return
     */
    private Set<ArrayList<PatternSplit>> generateConcatenatedSplit(ArrayList<PatternSplit> split) {
        Set<ArrayList<PatternSplit>> result = new HashSet<>();
        ArrayList<PatternSplit> currentConcatenation = new ArrayList<>();
        int length = split.get(0).getStartIndices().length;
        currentConcatenation.add(split.get(0));

        for (int i = 0; i < split.size() - 1; i++) {
            for (int j = 0; j < length; j++) {
                if (split.get(i).getStartIndices()[j] + split.get(i).getLengths()[j] != split.get(i + 1).getStartIndices()[j]) {
                    result.add(new ArrayList<>(currentConcatenation));
                    currentConcatenation = new ArrayList<>();
                    break;
                }
            }
            currentConcatenation.add(split.get(i + 1));
        }

        result.add(currentConcatenation);
        return  result;
    }


    /**
     * Sorts and orders the set of pattern splits based on their pattern node, device and chunk to execute.
     * @param step
     * @return
     */
    private Set<ArrayList<PatternSplit>> fuseSplits(StepMapping step) {
        Set<PatternSplit> splits = new HashSet<>(step.splits());
        HashMap<PatternNode, ArrayList<PatternSplit>> nodes = new HashMap<>();
        for (PatternSplit split: splits) {
            if (nodes.containsKey(split.getNode())) {
                nodes.get(split.getNode()).add(split);
            } else {
                ArrayList<PatternSplit> initializer = new ArrayList<>();
                initializer.add(split);
                nodes.put(split.getNode(), initializer);
            }
        }
        Set<ArrayList<PatternSplit>> result = new HashSet<>();
        for (ArrayList<PatternSplit> splitArray: nodes.values() ) {
            result.addAll(splitByDevice(step, splitArray));
        }
        result = splitByFusing(result);
        return result;
    }

    /**
     * Splits the set of given lists by the fused nodes. Thus only splits where all nodes are identical can remain within the same List.
     * @param currentSet
     * @return
     */
    private Set<ArrayList<PatternSplit>> splitByFusing(Set<ArrayList<PatternSplit>> currentSet) {
        Set<ArrayList<PatternSplit>> result = new HashSet<>();
        for (ArrayList<PatternSplit> current: currentSet) {
            result.addAll(singleSplitByFusing(current));
        }
        return result;
    }

    /**
     * Splits the list, such that for all fused nodes only identical pipelines on different data share the same list.
     * @param splits
     * @return
     */
    private Set<ArrayList<PatternSplit>> singleSplitByFusing(ArrayList<PatternSplit> splits) {
        Set<ArrayList<PatternSplit>> result = new HashSet<>();
        if (splits.size() <= 1) {
            result.add(splits);
            return result;
        }
        ArrayList<PatternSplit> subArray = new ArrayList<>();
        ArrayList<PatternSplit> remainder = new ArrayList<>();
        PatternSplit representative = splits.get(0);
        subArray.add(representative);
        for (int i = 1; i < splits.size(); i++) {
            PatternSplit toTest = splits.get(i);
            // Test for identical Parallel Pattern Splits
            if ( representative instanceof ParallelPatternSplit && toTest instanceof ParallelPatternSplit) {
                subArray.add(toTest);
                continue;
            }
            // Test for identical fused Pattern Splits
            if (representative instanceof FusedPatternSplit && toTest instanceof FusedPatternSplit) {
                if (((FusedPatternSplit) representative).getJobs().size() == ((FusedPatternSplit) toTest).getJobs().size()) {
                    boolean allStepsIdentical = true;
                    for (int j = 0; j < ((FusedPatternSplit) representative).getJobs().size(); j++) {
                        if (((FusedPatternSplit) representative).getJobs().get(j).getNode() != ((FusedPatternSplit) toTest).getJobs().get(j).getNode()) {
                            allStepsIdentical = false;
                        }
                    }
                    if (allStepsIdentical) {
                        subArray.add(toTest);
                        continue;
                    }
                }
            }
            remainder.add(toTest);
        }
        result.add(subArray);
        if (remainder.size() > 0) {
            result.addAll(singleSplitByFusing(remainder));
        }

        return result;

    }


    /**
     * Splits the list of pattern splits by their assigned device and sorts them.
     * @param step
     * @param unsorted
     * @return
     */
    private Set<ArrayList<PatternSplit>> splitByDevice(StepMapping step, ArrayList<PatternSplit> unsorted) {
        HashMap<Device, ArrayList<PatternSplit>> deviceMap = new HashMap<>();
        for (PatternSplit split : unsorted ) {
            if (deviceMap.containsKey(step.get(split).getDevice())) {
                deviceMap.get(step.get(split).getDevice()).add(split);
            } else {
                ArrayList<PatternSplit> initializer = new ArrayList<>();
                initializer.add(split);
                deviceMap.put(step.get(split).getDevice(), initializer);
            }
        }
        HashSet<ArrayList<PatternSplit>> result = new HashSet<>();
        for (ArrayList<PatternSplit> splits: deviceMap.values() ) {
            sortSplitsByStartIndex(splits);
            result.add(splits);
        }
        return result;
    }

    /**
     * Sorts an Array of pattern splits
     * @param splits
     */
    private void sortSplitsByStartIndex(ArrayList<PatternSplit> splits) {
        bubbleSort(splits);
    }

    /**
     * Implements a bubble sort on an array of pattern splits.
     * @param splits
     */
    private void bubbleSort(ArrayList<PatternSplit> splits) {
        for (int i = splits.size(); i > 1; i--) {
            for (int j = 0; j < i - 1; j++) {
                if (isGreater(splits.get(j).getStartIndices(), splits.get(j + 1).getStartIndices())) {
                    swap(splits, j, j + 1);
                }
            }
        }
    }

    /**
     * Switches the position of the element at position first and second in array splits.
     * @param splits
     * @param first
     * @param second
     */
    private void swap(ArrayList<PatternSplit> splits, int first, int second) {
        if (first >= splits.size() || second >= splits.size()) {
            Log.error("Array access to large for swap!");
            throw new RuntimeException("Critical error!");
        }
        PatternSplit temp = splits.get(first);
        splits.set(first, splits.get(second));
        splits.set(second, temp);
    }

    /**
     * Returns true if the array first is larger than the array second.
     * The comparison is based on the integer values on the same level, the values earlier are more important.
     * @param first
     * @param second
     * @return
     */
    private boolean isGreater(long[] first, long[] second) {
        for (int i = 0; i < first.length; i++) {
            if (first[i] > second[i]) {
                return true;
            } else if (first[i] < second[i]) {
                return false;
            }
        }
        return true;
    }


    private MappingNode generateMappingNode(PatternNode node, MappingNode parent, Optional<ParallelPatternSplit> split) {
        if (node instanceof BranchCaseNode) {
            return generateBranchCaseMapping((BranchCaseNode) node, parent);
        } else if (node instanceof BranchNode) {
            return generateBranchMapping((BranchNode) node, parent);
        } else if (node instanceof ParallelCallNode) {
            return generateSerializedParallelCallMapping((ParallelCallNode) node, parent, split);
        } else if (node instanceof CallNode) {
            return generateCallMapping((CallNode) node, parent);
        } else if (node instanceof ComplexExpressionNode) {
            return generateComplexExpressionMapping((ComplexExpressionNode) node, parent);
        } else if (node instanceof ForEachLoopNode) {
            return generateForEachLoopMapping((ForEachLoopNode) node, parent);
        } else if (node instanceof ForLoopNode) {
            return generateForLoopMapping((ForLoopNode) node, parent);
        } else if (node instanceof ReturnNode) {
            return generateReturnMapping((ReturnNode) node, parent);
        } else if (node instanceof SimpleExpressionBlockNode) {
            return generateSimpleExpressionBlockMapping((SimpleExpressionBlockNode) node, parent);
        } else if (node instanceof WhileLoopNode) {
            return generateWhileLoopMapping((WhileLoopNode) node, parent);
        } else if (node instanceof LoopSkipNode) {
            return generateLoopSkipMapping((LoopSkipNode) node, parent);
        } else if (node instanceof JumpStatementNode) {
            return generateJumpStatementMapping((JumpStatementNode) node, parent);
        } else if (node instanceof JumpLabelNode) {
            return generateJumpLabelMapping((JumpLabelNode) node, parent);
        } else {
            Log.error("Pattern type not recognized");
            throw new RuntimeException("Critical error!");
        }
    }

    private JumpStatementMapping generateJumpStatementMapping(JumpStatementNode node, MappingNode parent) {
        JumpStatementMapping result = new JumpStatementMapping(Optional.of(parent), node.getVariableTable(), node, node.getClosingVars(), node.getLabel(), node.getOutputData(), AbstractMappingTree.getDefaultDevice().getParent());
        ComplexExpressionMapping resultExpression = generateComplexExpressionMapping(node.getResultExpression(), result);
        result.setResultExpression(resultExpression);
        return result;
    }

    private JumpLabelMapping generateJumpLabelMapping(JumpLabelNode node, MappingNode parent) {
        JumpLabelMapping result = new JumpLabelMapping(Optional.of(parent), node.getVariableTable(), node, node.getLabel(), AbstractMappingTree.getDefaultDevice().getParent());
        return result;
    }

    /**
     * Generates the AMT node for the LoopSkip
     * @param node
     * @param parent
     * @return
     */
    private LoopSkipMapping generateLoopSkipMapping(LoopSkipNode node, MappingNode parent) {
        return new LoopSkipMapping(Optional.of(parent), node.getVariableTable(), node, AbstractMappingTree.getDefaultDevice().getParent());
    }

    /**
     * Generates a serial version of a parallel pattern call
     * @param node
     * @param parent
     * @param splitOPT
     * @return
     */
    private SerializedParallelCallMapping generateSerializedParallelCallMapping(ParallelCallNode node, MappingNode parent, Optional<ParallelPatternSplit> splitOPT) {
        ArrayList<Long> iterations = new ArrayList<>();
        ArrayList<Long> starts = new ArrayList<>();
        if (splitOPT.isPresent()) {
            ParallelPatternSplit split = splitOPT.get();

            int length = Integer.min(split.getStartIndices().length, split.getLengths().length);
            for (int i = 0; i < length; i++) {
                iterations.add(split.getLengths()[i]);
            }

            for (long start : split.getStartIndices()) {
                starts.add(start);
            }
        } else {
            // Generate the start and iteration counts based on the original APT.
            // The position of the values within the additional arguments is defined by the language.
            FunctionNode function = AbstractPatternTree.getFunctionTable().get(node.getFunctionIdentifier());
            if (function instanceof MapNode) {
                starts.add(((MetaValue<Long>) node.getAdditionalArguments().get(1)).getValue());
                iterations.add(((MetaValue<Long>) node.getAdditionalArguments().get(0)).getValue());
            } else if (function instanceof ReduceNode) {
                starts.add(((MetaList<Long>) node.getAdditionalArguments().get(0)).getValues().get(3));
                iterations.add(((MetaList<Long>) node.getAdditionalArguments().get(0)).getValues().get(0));
            } else if (function instanceof StencilNode) {
                starts.addAll(((MetaList<Long>) node.getAdditionalArguments().get(1)).getValues());
                iterations.addAll(((MetaList<Long>) node.getAdditionalArguments().get(0)).getValues());
            } else if (function instanceof DynamicProgrammingNode) {
                starts.addAll(((MetaList<Long>) node.getAdditionalArguments().get(2)).getValues());
                iterations.add(((MetaValue<Long>) node.getAdditionalArguments().get(0)).getValue());
                iterations.add(((MetaValue<Long>) node.getAdditionalArguments().get(1)).getValue());
            }
        }
        SerializedParallelCallMapping result = new SerializedParallelCallMapping(Optional.of(parent), node.getVariableTable(), node, starts, iterations, AbstractMappingTree.getDefaultDevice().getProcessor().get(0), 0);
        result.setChildren(new ArrayList<>());
        result.setDefinition(generateComplexExpressionMapping((ComplexExpressionNode) node.getChildren().get(0), result));

        return result;
    }


    private ComplexExpressionMapping generateComplexExpressionMapping(ComplexExpressionNode node, MappingNode parent) {
        ComplexExpressionMapping mapping = new ComplexExpressionMapping(Optional.of(parent), node.getVariableTable(), node, AbstractMappingTree.getDefaultDevice().getParent());
        ArrayList<MappingNode> children = new ArrayList<>();
        for (PatternNode child: node.getChildren() ) {
            children.add(generateMappingNode(child, parent, Optional.empty()));
        }
        mapping.setChildren(children);
        return mapping;
    }


    private CallMapping generateCallMapping(CallNode node, MappingNode parent) {
        return new CallMapping(Optional.of(parent), node.getVariableTable(), node, AbstractMappingTree.getDefaultDevice().getParent());
    }

    private SimpleExpressionBlockMapping generateSimpleExpressionBlockMapping(SimpleExpressionBlockNode node, MappingNode parent) {
        return new SimpleExpressionBlockMapping(Optional.of(parent), node.getVariableTable(), node, AbstractMappingTree.getDefaultDevice().getParent());
    }

    private WhileLoopMapping generateWhileLoopMapping(WhileLoopNode node, MappingNode parent) {
        WhileLoopMapping mapping = new WhileLoopMapping(Optional.of(parent), node.getVariableTable(), node, AbstractMappingTree.getDefaultDevice().getParent());
        ArrayList<MappingNode> children = new ArrayList<>();
        for (int i = 1; i < node.getChildren().size(); i++) {
            children.add(generateMappingNode(node.getChildren().get(i), mapping, Optional.empty()));
        }
        mapping.setChildren(children);
        mapping.setCondition(generateComplexExpressionMapping((ComplexExpressionNode) node.getChildren().get(0), mapping));
        return mapping;
    }

    private ReturnMapping generateReturnMapping(ReturnNode node, MappingNode parent) {
        ReturnMapping mapping = new ReturnMapping(Optional.of(parent), node.getVariableTable(), node, AbstractMappingTree.getDefaultDevice().getParent());
        if (node.getChildren().size() == 1) {
            mapping.setResult(generateComplexExpressionMapping((ComplexExpressionNode) node.getChildren().get(0), mapping));
        }
        mapping.setChildren(new ArrayList<>());
        return mapping;
    }

    private ForLoopMapping generateForLoopMapping(ForLoopNode node, MappingNode parent) {
        ForLoopMapping mapping = new ForLoopMapping(Optional.of(parent), node.getVariableTable(), node, AbstractMappingTree.getDefaultDevice().getParent());
        mapping.setInitExpression(generateComplexExpressionMapping((ComplexExpressionNode) node.getChildren().get(0), mapping));
        mapping.setControlExpression(generateComplexExpressionMapping((ComplexExpressionNode) node.getChildren().get(1), mapping));
        mapping.setUpdateExpression(generateComplexExpressionMapping((ComplexExpressionNode) node.getChildren().get(2), mapping));

        ArrayList<MappingNode> children = new ArrayList<>();
        for (int i = 3; i < node.getChildren().size(); i++) {
            children.add(generateMappingNode(node.getChildren().get(i), mapping, Optional.empty()));
        }
        mapping.setChildren(children);

        return mapping;
    }


    private ForEachLoopMapping generateForEachLoopMapping(ForEachLoopNode node, MappingNode parent) {
        ForEachLoopMapping mapping = new ForEachLoopMapping(Optional.of(parent), node.getVariableTable(), node, AbstractMappingTree.getDefaultDevice().getParent());

        mapping.setParsedList(generateComplexExpressionMapping((ComplexExpressionNode) node.getChildren().get(0), mapping));

        ArrayList<MappingNode> children = new ArrayList<>();
        for (int i = 1; i < node.getChildren().size(); i++) {
            children.add(generateMappingNode(node.getChildren().get(i), mapping, Optional.empty()));
        }

        mapping.setChildren(children);
        return mapping;
    }

    private BranchMapping generateBranchMapping(BranchNode node, MappingNode parent) {
        BranchMapping mapping = new BranchMapping(Optional.of(parent), node.getVariableTable(), node, AbstractMappingTree.getDefaultDevice().getParent());
        ArrayList<MappingNode> children = new ArrayList<>();
        for (PatternNode child: node.getChildren() ) {
            children.add(generateMappingNode(child, mapping, Optional.empty()));
        }
        mapping.setChildren(children);
        return mapping;
    }

    private BranchCaseMapping generateBranchCaseMapping(BranchCaseNode node, MappingNode parent) {
        BranchCaseMapping mapping = new BranchCaseMapping(Optional.of(parent), node.getVariableTable(), node, AbstractMappingTree.getDefaultDevice().getParent());
        ArrayList<MappingNode> children = new ArrayList<>();
        if (node.isHasCondition()) {
            mapping.setCondition(Optional.of(generateComplexExpressionMapping((ComplexExpressionNode) node.getChildren().get(0), mapping)));
            for (int i = 1; i < node.getChildren().size(); i++) {
                children.add(generateMappingNode(node.getChildren().get(i), mapping, Optional.empty()));
            }
        } else {
            mapping.setCondition(Optional.empty());
            for (PatternNode child: node.getChildren() ) {
                children.add(generateMappingNode(child, parent, Optional.empty()));
            }
        }
        mapping.setChildren(children);

        return mapping;
    }
}
