package de.parallelpatterndsl.patterndsl.MappingTree;

import de.parallelpatterndsl.patterndsl.MappingTree.Nodes.DataControl.BarrierMapping;
import de.parallelpatterndsl.patterndsl.MappingTree.Nodes.Function.*;
import de.parallelpatterndsl.patterndsl.MappingTree.Nodes.FunctionMapping;
import de.parallelpatterndsl.patterndsl.MappingTree.Nodes.MappingNode;
import de.parallelpatterndsl.patterndsl.MappingTree.Nodes.ParallelCalls.GPUParallelCallMapping;
import de.parallelpatterndsl.patterndsl.MappingTree.Nodes.ParallelCalls.ParallelCallMapping;
import de.parallelpatterndsl.patterndsl.MappingTree.Nodes.ParallelCalls.ReductionCallMapping;
import de.parallelpatterndsl.patterndsl.MappingTree.Nodes.ParallelCalls.SerializedParallelCallMapping;
import de.parallelpatterndsl.patterndsl.MappingTree.Nodes.Plain.*;
import de.parallelpatterndsl.patterndsl.abstractPatternTree.AbstractPatternTree;
import de.parallelpatterndsl.patterndsl.abstractPatternTree.DataElements.TempData;
import de.parallelpatterndsl.patterndsl.abstractPatternTree.Nodes.FunctionNode;
import de.parallelpatterndsl.patterndsl.abstractPatternTree.Nodes.Functions.*;
import de.parallelpatterndsl.patterndsl.abstractPatternTree.Nodes.PatternNode;
import de.parallelpatterndsl.patterndsl.abstractPatternTree.Nodes.Plain.*;
import de.parallelpatterndsl.patterndsl.abstractPatternTree.Nodes.Plain.AdditionalArguments.MetaList;
import de.parallelpatterndsl.patterndsl.abstractPatternTree.Nodes.Plain.AdditionalArguments.MetaValue;
import de.parallelpatterndsl.patterndsl.hardwareDescription.Network;
import de.parallelpatterndsl.patterndsl.hardwareDescription.Processor;
import de.parallelpatterndsl.patterndsl.helperLibrary.RandomStringGenerator;
import de.se_rwth.commons.logging.Log;

import java.util.*;

public class DebugCreator {

    private AbstractPatternTree APT;

    private Network network;

    private boolean isNested = true;

    private HashMap<String, FunctionMapping> functionTable = new HashMap<>();

    public DebugCreator(AbstractPatternTree APT, Network network) {
        this.APT = APT;
        this.network = network;
    }

    public AbstractMappingTree generate() {
        for (FunctionNode node : AbstractPatternTree.getFunctionTable().values() ) {
            if (node.isAvailableAfterInlining() || node.getIdentifier().equals("main")) {
                functionTable.put(node.getIdentifier(), generateAMT(node));
            }
        }
        AbstractMappingTree.setFunctionTable(functionTable);
        AbstractMappingTree AMT = new AbstractMappingTree((MainMapping) AbstractMappingTree.getFunctionTable().get("main"), APT.getGlobalVariableTable(), APT.getGlobalAssignments());
        return AMT;
    }


    public FunctionMapping generateAMT(FunctionNode node) {

        boolean isMain = false;
        FunctionMapping functionMapping;
        if (node instanceof MapNode) {
            functionMapping = new MapMapping((MapNode) node);
        } else if (node instanceof DynamicProgrammingNode) {
            functionMapping = new DynamicProgrammingMapping((DynamicProgrammingNode) node);
        } else if (node instanceof MainNode) {
            functionMapping = new MainMapping((MainNode) node);
            isMain = true;
            isNested = false;
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

        ArrayList<MappingNode> children = new ArrayList<>();
        for (PatternNode child : node.getChildren()) {
            children.addAll(generateMappingNode(child, Optional.of(functionMapping)));
        }


        functionMapping.setChildren(children);

        isNested = true;
        return functionMapping;
    }


    private ArrayList<MappingNode> generateMappingNode(PatternNode node, Optional<MappingNode> parent) {
        ArrayList<MappingNode> res = new ArrayList<>();
        if (node instanceof BranchCaseNode) {
            res.add(generateBranchCaseMapping( parent, (BranchCaseNode) node));
        } else if (node instanceof BranchNode) {
            res.add(generateBranchMapping( parent, (BranchNode) node));
        } else if (node instanceof ComplexExpressionNode) {
            res.add(generateComplexExpressionMapping( parent, (ComplexExpressionNode) node));
        } else if (node instanceof ParallelCallNode) {
            res.addAll(generateParallelCalls( parent, (ParallelCallNode) node));
        } else if (node instanceof ForEachLoopNode) {
            res.add(generateForEachLoopMapping( parent, (ForEachLoopNode) node));
        } else if (node instanceof ForLoopNode) {
            res.add(generateForLoopMapping( parent, (ForLoopNode) node));
        } else if (node instanceof JumpLabelNode) {
            res.add(generateJumpLabelMapping( parent, (JumpLabelNode) node));
        } else if (node instanceof JumpStatementNode) {
            res.add(generateJumpStatementMapping( parent, (JumpStatementNode) node));
        } else if (node instanceof ReturnNode) {
            res.add(generateReturnMapping( parent, (ReturnNode) node));
        } else if (node instanceof SimpleExpressionBlockNode) {
            res.add(generateSimpleExpressionBlockMapping( parent, (SimpleExpressionBlockNode) node));
        } else if (node instanceof WhileLoopNode) {
            res.add(generateWhileLoopMapping( parent, (WhileLoopNode) node));
        } else if (node instanceof CallNode) {
            res.add(generateCallMapping( parent, (CallNode) node));
        } else {
            Log.error("Node type not defined! ");
            throw new RuntimeException("Critical error!");
        }

        return res;

    }


    private BranchCaseMapping generateBranchCaseMapping(Optional<MappingNode> parent, BranchCaseNode node) {
        BranchCaseMapping res = new BranchCaseMapping(parent, node.getVariableTable(), node, AbstractMappingTree.getDefaultDevice().getParent());
        if (node.isHasCondition()) {
            ComplexExpressionMapping condition = generateComplexExpressionMapping(Optional.of(res), (ComplexExpressionNode) node.getChildren().get(0));
            res.setCondition(Optional.of(condition));
            ArrayList<MappingNode> children = new ArrayList<>();
            for (int i = 1; i < node.getChildren().size(); i++) {
                children.addAll(generateMappingNode(node.getChildren().get(i), Optional.of(res)));
            }
            res.setChildren(children);
        } else {
            res.setCondition(Optional.empty());
            ArrayList<MappingNode> children = new ArrayList<>();
            for (PatternNode child : node.getChildren() ) {
                children.addAll(generateMappingNode(child, Optional.of(res)));
            }
            res.setChildren(children);
        }
        return res;
    }

    private ComplexExpressionMapping generateComplexExpressionMapping(Optional<MappingNode> parent, ComplexExpressionNode node) {
        ComplexExpressionMapping res = new ComplexExpressionMapping(parent, node.getVariableTable(), node, AbstractMappingTree.getDefaultDevice().getParent());

        ArrayList<MappingNode> children = new ArrayList<>();
        for (PatternNode child : node.getChildren() ) {
            children.addAll(generateMappingNode(child, Optional.of(res)));
        }
        res.setChildren(children);

        return res;
    }

    private BranchMapping generateBranchMapping(Optional<MappingNode> parent, BranchNode node) {
        BranchMapping res = new BranchMapping(parent, node.getVariableTable(), node, AbstractMappingTree.getDefaultDevice().getParent());
        ArrayList<MappingNode> children = new ArrayList<>();
        for (PatternNode child : node.getChildren() ) {
            children.addAll(generateMappingNode(child, Optional.of(res)));
        }
        res.setChildren(children);

        return res;
    }

    private CallMapping generateCallMapping(Optional<MappingNode> parent, CallNode node) {
        CallMapping res = new CallMapping(parent, node.getVariableTable(), node, AbstractMappingTree.getDefaultDevice().getParent());

        ArrayList<MappingNode> children = new ArrayList<>();
        res.setChildren(children);

        return res;
    }

    private ForEachLoopMapping generateForEachLoopMapping(Optional<MappingNode> parent, ForEachLoopNode node) {
        ForEachLoopMapping res = new ForEachLoopMapping(parent, node.getVariableTable(), node, AbstractMappingTree.getDefaultDevice().getParent());

        ArrayList<MappingNode> children = new ArrayList<>();
        for (int i = 1; i < node.getChildren().size(); i++) {
            children.addAll(generateMappingNode(node.getChildren().get(i), Optional.of(res)));
        }
        res.setChildren(children);

        res.setParsedList(generateComplexExpressionMapping(Optional.of(res), (ComplexExpressionNode) node.getChildren().get(0)));

        return res;
    }

    private ForLoopMapping generateForLoopMapping(Optional<MappingNode> parent, ForLoopNode node) {
        ForLoopMapping res = new ForLoopMapping(parent, node.getVariableTable(), node, AbstractMappingTree.getDefaultDevice().getParent());

        ArrayList<MappingNode> children = new ArrayList<>();
        for (int i = 3; i < node.getChildren().size(); i++) {
            children.addAll(generateMappingNode(node.getChildren().get(i), Optional.of(res)));
        }
        res.setChildren(children);

        res.setInitExpression(generateComplexExpressionMapping(Optional.of(res), (ComplexExpressionNode) node.getChildren().get(0)));
        res.setControlExpression(generateComplexExpressionMapping(Optional.of(res), (ComplexExpressionNode) node.getChildren().get(1)));
        res.setUpdateExpression(generateComplexExpressionMapping(Optional.of(res), (ComplexExpressionNode) node.getChildren().get(2)));

        return res;
    }


    private JumpLabelMapping generateJumpLabelMapping(Optional<MappingNode> parent, JumpLabelNode node) {
        JumpLabelMapping res = new JumpLabelMapping(parent, node.getVariableTable(), node, node.getLabel(), AbstractMappingTree.getDefaultDevice().getParent());

        ArrayList<MappingNode> children = new ArrayList<>();
        for (PatternNode child : node.getChildren() ) {
            children.addAll(generateMappingNode(child, Optional.of(res)));
        }
        res.setChildren(children);

        return res;
    }

    private JumpStatementMapping generateJumpStatementMapping(Optional<MappingNode> parent, JumpStatementNode node) {
        JumpStatementMapping res = new JumpStatementMapping(parent, node.getVariableTable(), node, node.getClosingVars(), generateComplexExpressionMapping(Optional.empty(), node.getResultExpression()), node.getLabel(), node.getOutputData(), AbstractMappingTree.getDefaultDevice().getParent());
        res.getResultExpression().setParent(Optional.of(res));

        ArrayList<MappingNode> children = new ArrayList<>();
        for (PatternNode child : node.getChildren() ) {
            children.addAll(generateMappingNode(child, Optional.of(res)));
        }
        res.setChildren(children);

        return res;
    }

    private ReturnMapping generateReturnMapping(Optional<MappingNode> parent, ReturnNode node) {
        ReturnMapping res = new ReturnMapping(parent, node.getVariableTable(), node, AbstractMappingTree.getDefaultDevice().getParent());

        ArrayList<MappingNode> children = new ArrayList<>();
        for (int i = 1; i < node.getChildren().size(); i++) {
            children.addAll(generateMappingNode(node.getChildren().get(i), Optional.of(res)));
        }
        res.setChildren(children);

        res.setResult(generateComplexExpressionMapping(Optional.of(res), (ComplexExpressionNode) node.getChildren().get(0)));

        return res;
    }

    private WhileLoopMapping generateWhileLoopMapping(Optional<MappingNode> parent, WhileLoopNode node) {
        WhileLoopMapping res = new WhileLoopMapping(parent, node.getVariableTable(), node, AbstractMappingTree.getDefaultDevice().getParent());

        ArrayList<MappingNode> children = new ArrayList<>();
        for (int i = 1; i < node.getChildren().size(); i++) {
            children.addAll(generateMappingNode(node.getChildren().get(i), Optional.of(res)));
        }
        res.setChildren(children);

        res.setCondition(generateComplexExpressionMapping(Optional.of(res), (ComplexExpressionNode) node.getChildren().get(0)));

        return res;
    }

    private SimpleExpressionBlockMapping generateSimpleExpressionBlockMapping(Optional<MappingNode> parent, SimpleExpressionBlockNode node) {
        SimpleExpressionBlockMapping res = new SimpleExpressionBlockMapping(parent, node.getVariableTable(), node, AbstractMappingTree.getDefaultDevice().getParent());

        ArrayList<MappingNode> children = new ArrayList<>();
        for (PatternNode child : node.getChildren() ) {
            children.addAll(generateMappingNode(child, Optional.of(res)));
        }
        res.setChildren(children);

        return res;
    }


    private ArrayList<MappingNode> generateParallelCalls(Optional<MappingNode> parent, ParallelCallNode node) {
        ArrayList<MappingNode> res = new ArrayList<>();
        FunctionNode functionNode = AbstractPatternTree.getFunctionTable().get(node.getFunctionIdentifier());

        ArrayList<Long> starts = new ArrayList<>();
        ArrayList<Long> numIterations = new ArrayList<>();

        ArrayList<ArrayList<Long>> splitStarts = new ArrayList<>();
        ArrayList<ArrayList<Long>> splitIterations = new ArrayList<>();

        ArrayList<Processor> executionUnits = new ArrayList<>();

        if (functionNode instanceof MapNode) {
            MetaValue metaStart = (MetaValue) node.getAdditionalArguments().get(1);
            starts.add((Long) metaStart.getValue());
            MetaValue metaLength = (MetaValue) node.getAdditionalArguments().get(0);
            numIterations.add((Long) metaLength.getValue());
            for (int i = 2; i < node.getAdditionalArguments().size(); i++) {
                MetaList<Integer> metaExecutor = (MetaList<Integer>) node.getAdditionalArguments().get(i);
                executionUnits.add(network.getNodes().get(metaExecutor.getValues().get(0)).getDevices().get(metaExecutor.getValues().get(1)).getProcessor().get(metaExecutor.getValues().get(2)));
            }


            if (executionUnits.size() > 1) {
                for (int i = 0; i < executionUnits.size(); i++) {
                    long chunkSize = numIterations.get(0) / executionUnits.size();
                    if (numIterations.get(0) % executionUnits.size() > i) {
                        chunkSize++;
                    }
                    ArrayList<Long> startsplit = new ArrayList<>();
                    ArrayList<Long> numIterationsplit = new ArrayList<>();
                    if (i == 0){
                        startsplit.add(starts.get(0));
                    } else {
                        startsplit.add(splitStarts.get(i - 1).get(0) + splitIterations.get(i - 1).get(0));
                    }
                    numIterationsplit.add(chunkSize);
                    splitStarts.add(startsplit);
                    splitIterations.add(numIterationsplit);
                }
            } else {
                splitIterations.add(numIterations);
                splitStarts.add(starts);
            }

            if (!isNested) {
                for (int i = 0; i < executionUnits.size(); i++) {
                    if (executionUnits.get(i).getParent().getType().equals("GPU")) {
                        GPUParallelCallMapping gpuParallelCallMapping = new GPUParallelCallMapping(parent, node.getVariableTable(), node, splitStarts.get(i), splitIterations.get(i), executionUnits.get(i), executionUnits.get(i).getCores(), Optional.empty(), new HashSet<>(), 1);
                        res.add(gpuParallelCallMapping);
                        gpuParallelCallMapping.setDefinition(generateComplexExpressionMapping(Optional.of(gpuParallelCallMapping), (ComplexExpressionNode) node.getChildren().get(0)));
                    } else if (executionUnits.get(i).getParent().getType().equals("CPU")) {
                        ParallelCallMapping parallelCallMapping = new ParallelCallMapping(parent, node.getVariableTable(), node, splitStarts.get(i), splitIterations.get(i), executionUnits.get(i), executionUnits.get(i).getCores(), Optional.empty(), new HashSet<>());

                        res.add(parallelCallMapping);
                        parallelCallMapping.setDefinition(generateComplexExpressionMapping(Optional.of(parallelCallMapping), (ComplexExpressionNode) node.getChildren().get(0)));
                    }

                }
            }


        } else if (functionNode instanceof ReduceNode) {
            MetaList<Long> metaList = (MetaList<Long>) node.getAdditionalArguments().get(0);
            numIterations.add(metaList.getValues().get(0));
            starts.add(metaList.getValues().get(3));
            for (int i = 1; i < node.getAdditionalArguments().size(); i++) {
                MetaList<Integer> metaExecutor = (MetaList<Integer>) node.getAdditionalArguments().get(i);
                executionUnits.add(network.getNodes().get(metaExecutor.getValues().get(0)).getDevices().get(metaExecutor.getValues().get(1)).getProcessor().get(metaExecutor.getValues().get(2)));
            }

            if (executionUnits.size() > 1) {
                for (int i = 0; i < executionUnits.size(); i++) {
                    long chunkSize = numIterations.get(0) / executionUnits.size();
                    if (numIterations.get(0) % executionUnits.size() > i) {
                        chunkSize++;
                    }
                    ArrayList<Long> startsplit = new ArrayList<>();
                    ArrayList<Long> numIterationsplit = new ArrayList<>();
                    if (i == 0){
                        startsplit.add(starts.get(0));
                    } else {
                        startsplit.add(splitStarts.get(i - 1).get(0) + splitIterations.get(i - 1).get(0));
                    }
                    numIterationsplit.add(chunkSize);
                    splitStarts.add(startsplit);
                    splitIterations.add(numIterationsplit);
                }
            } else {
                splitIterations.add(numIterations);
                splitStarts.add(starts);
            }



            if (!isNested) {
                if (executionUnits.size() > 1) {
                    HashSet<TempData> tempDatas = new HashSet<>();
                    for (int i = 0; i < executionUnits.size(); i++) {
                        TempData tempData = new TempData(((ReduceNode) functionNode).getReturnElement().getTypeName(), "TempData_ " + RandomStringGenerator.getAlphaNumericString());
                        tempDatas.add(tempData);
                        HashSet<TempData> wrapper = new HashSet<>();
                        wrapper.add(tempData);
                        ReductionCallMapping reductionCallMapping = new ReductionCallMapping(parent, node.getVariableTable(), node, splitStarts.get(i), splitIterations.get(i), executionUnits.get(i), executionUnits.get(i).getCores(), false, new HashSet<>(), wrapper, 1, executionUnits.get(i).getParent().getType().equals("GPU"));
                        res.add(reductionCallMapping);
                        reductionCallMapping.setDefinition(generateComplexExpressionMapping(Optional.of(reductionCallMapping), (ComplexExpressionNode) node.getChildren().get(0)));
                    }

                    res.add(new BarrierMapping(parent, node.getVariableTable(), new HashSet<>(executionUnits)));

                    ReductionCallMapping reductionCallMapping = new ReductionCallMapping(parent, node.getVariableTable(), node, starts, numIterations, executionUnits.get(0), 1, true, tempDatas, new HashSet<>(), 1, executionUnits.get(0).getParent().getType().equals("GPU"));

                    res.add(reductionCallMapping);
                    reductionCallMapping.setDefinition(generateComplexExpressionMapping(Optional.of(reductionCallMapping), (ComplexExpressionNode) node.getChildren().get(0)));
                } else {
                    ReductionCallMapping reductionCallMapping = new ReductionCallMapping(parent, node.getVariableTable(), node, starts, numIterations, executionUnits.get(0), executionUnits.get(0).getCores(), false, new HashSet<>(), new HashSet<>(), 1, executionUnits.get(0).getParent().getType().equals("GPU"));

                    res.add(reductionCallMapping);
                    reductionCallMapping.setDefinition(generateComplexExpressionMapping(Optional.of(reductionCallMapping), (ComplexExpressionNode) node.getChildren().get(0)));
                }
            }

        } else if (functionNode instanceof StencilNode) {
            MetaList<Long> metaIterations = (MetaList<Long>) node.getAdditionalArguments().get(0);
            numIterations.addAll(metaIterations.getValues());
            MetaList<Long> metastarts = (MetaList<Long>) node.getAdditionalArguments().get(1);
            starts.addAll(metastarts.getValues());
            for (int i = 2; i < node.getAdditionalArguments().size(); i++) {
                MetaList<Integer> metaExecutor = (MetaList<Integer>) node.getAdditionalArguments().get(i);
                executionUnits.add(network.getNodes().get(metaExecutor.getValues().get(0)).getDevices().get(metaExecutor.getValues().get(1)).getProcessor().get(metaExecutor.getValues().get(2)));
            }

            if (executionUnits.size() > 1) {
                for (int i = 0; i < executionUnits.size(); i++) {
                    long chunkSize = numIterations.get(0) / executionUnits.size();
                    if (numIterations.get(0) % executionUnits.size() > i) {
                        chunkSize++;
                    }
                    ArrayList<Long> startsplit = new ArrayList<>();
                    ArrayList<Long> numIterationsplit = new ArrayList<>();
                    if (i == 0){
                        startsplit.add(starts.get(0));
                        for (int j = 1; j < starts.size(); j++) {
                            startsplit.add(starts.get(j));
                        }
                    } else {
                        startsplit.add(splitStarts.get(i - 1).get(0) + splitIterations.get(i - 1).get(0));
                        for (int j = 1; j < starts.size(); j++) {
                            startsplit.add(starts.get(j));
                        }
                    }

                    numIterationsplit.add(chunkSize);
                    for (int j = 1; j < numIterations.size(); j++) {
                        numIterationsplit.add(numIterations.get(j));
                    }


                    splitStarts.add(startsplit);
                    splitIterations.add(numIterationsplit);
                }
            } else {
                splitIterations.add(numIterations);
                splitStarts.add(starts);
            }

            if (!isNested) {
                for (int i = 0; i < executionUnits.size(); i++) {
                    if (executionUnits.get(i).getParent().getType().equals("GPU")) {
                        GPUParallelCallMapping gpuParallelCallMapping = new GPUParallelCallMapping(parent, node.getVariableTable(), node, splitStarts.get(i), splitIterations.get(i), executionUnits.get(i), executionUnits.get(i).getCores(), Optional.empty(), new HashSet<>(), 1);
                        res.add(gpuParallelCallMapping);
                        gpuParallelCallMapping.setDefinition(generateComplexExpressionMapping(Optional.of(gpuParallelCallMapping), (ComplexExpressionNode) node.getChildren().get(0)));
                    } else if (executionUnits.get(i).getParent().getType().equals("CPU")) {
                        ParallelCallMapping parallelCallMapping = new ParallelCallMapping(parent, node.getVariableTable(), node, splitStarts.get(i), splitIterations.get(i), executionUnits.get(i), executionUnits.get(i).getCores(), Optional.empty(), new HashSet<>());
                        res.add(parallelCallMapping);
                        parallelCallMapping.setDefinition(generateComplexExpressionMapping(Optional.of(parallelCallMapping), (ComplexExpressionNode) node.getChildren().get(0)));
                    }

                }
            }
        } else if (functionNode instanceof DynamicProgrammingNode) {
            MetaValue metaTimes = (MetaValue) node.getAdditionalArguments().get(0);
            numIterations.add((Long) metaTimes.getValue());
            MetaValue metaLength = (MetaValue) node.getAdditionalArguments().get(1);
            numIterations.add((Long) metaLength.getValue());
            MetaList<Long> metaList = (MetaList<Long>) node.getAdditionalArguments().get(2);
            starts.add(metaList.getValues().get(0));
            starts.add(metaList.getValues().get(1));
            for (int i = 3; i < node.getAdditionalArguments().size(); i++) {
                MetaList<Integer> metaExecutor = (MetaList<Integer>) node.getAdditionalArguments().get(i);
                executionUnits.add(network.getNodes().get(metaExecutor.getValues().get(0)).getDevices().get(metaExecutor.getValues().get(1)).getProcessor().get(metaExecutor.getValues().get(2)));
            }

            if (!isNested) {


                if (executionUnits.size() > 1) {
                    for (int i = 0; i < executionUnits.size(); i++) {
                        long chunkSize = numIterations.get(1) / executionUnits.size();
                        if (numIterations.get(1) % executionUnits.size() > i) {
                            chunkSize++;
                        }
                        ArrayList<Long> startsplit = new ArrayList<>();
                        ArrayList<Long> numIterationsplit = new ArrayList<>();
                        if (i == 0) {
                            startsplit.add(starts.get(0));
                            startsplit.add(starts.get(1));
                        } else {
                            startsplit.add(starts.get(0));
                            startsplit.add(splitStarts.get(i - 1).get(1) + splitIterations.get(i - 1).get(1));
                        }

                        numIterationsplit.add(numIterations.get(0));
                        numIterationsplit.add(chunkSize);

                        splitStarts.add(startsplit);
                        splitIterations.add(numIterationsplit);
                    }
                } else {
                    splitIterations.add(numIterations);
                    splitStarts.add(starts);
                }



                for (int i = 0; i < executionUnits.size(); i++) {
                    if (executionUnits.get(i).getParent().getType().equals("GPU")) {
                        GPUParallelCallMapping gpuParallelCallMapping = new GPUParallelCallMapping(parent, node.getVariableTable(), node, splitStarts.get(i), splitIterations.get(i), executionUnits.get(i), executionUnits.get(i).getCores(), Optional.empty(), new HashSet<>(), 1);
                        res.add(gpuParallelCallMapping);
                        gpuParallelCallMapping.setDefinition(generateComplexExpressionMapping(Optional.of(gpuParallelCallMapping), (ComplexExpressionNode) node.getChildren().get(0)));
                    } else if (executionUnits.get(i).getParent().getType().equals("CPU")) {
                        ParallelCallMapping parallelCallMapping = new ParallelCallMapping(parent, node.getVariableTable(), node, splitStarts.get(i), splitIterations.get(i), executionUnits.get(i), executionUnits.get(i).getCores(), Optional.empty(), new HashSet<>());
                        res.add(parallelCallMapping);
                        parallelCallMapping.setDefinition(generateComplexExpressionMapping(Optional.of(parallelCallMapping), (ComplexExpressionNode) node.getChildren().get(0)));
                    }

                }
            }
        } else if (functionNode instanceof RecursionNode) {
            for (int i = 3; i < node.getAdditionalArguments().size(); i++) {
                MetaList<Integer> metaExecutor = (MetaList<Integer>) node.getAdditionalArguments().get(i);
                executionUnits.add(network.getNodes().get(metaExecutor.getValues().get(0)).getDevices().get(metaExecutor.getValues().get(1)).getProcessor().get(metaExecutor.getValues().get(2)));
            }
            starts.add((long) 0);
            numIterations.add((long) 1);
            ParallelCallMapping recursion = new ParallelCallMapping(parent, node.getVariableTable(), node, starts, numIterations, network.getNodes().get(0).getDevices().get(0).getProcessor().get(0), 1, Optional.empty(), new HashSet<>());
            res.add(recursion);
            recursion.setDefinition(generateComplexExpressionMapping(Optional.of(recursion), (ComplexExpressionNode) node.getChildren().get(0)));
            return res;
        }



        if (isNested) {
            SerializedParallelCallMapping serializedParallelCallMapping = new SerializedParallelCallMapping(parent, node.getVariableTable(), node,starts, numIterations, network.getNodes().get(0).getDevices().get(0).getProcessor().get(0), 1 );
            serializedParallelCallMapping.setDefinition(generateComplexExpressionMapping(Optional.of(serializedParallelCallMapping), (ComplexExpressionNode) node.getChildren().get(0)));
            res.add(serializedParallelCallMapping);
            return res;
        }



        return res;


    }


}
