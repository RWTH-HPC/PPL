package de.parallelpatterndsl.patterndsl.MappingTree.JSONReader;

import de.parallelpatterndsl.patterndsl.JSONPrinter.NodeIDMapping;
import de.parallelpatterndsl.patterndsl.MappingTree.AbstractMappingTree;
import de.parallelpatterndsl.patterndsl.MappingTree.Nodes.Function.*;
import de.parallelpatterndsl.patterndsl.MappingTree.Nodes.FunctionMapping;
import de.parallelpatterndsl.patterndsl.MappingTree.Nodes.MappingNode;
import de.parallelpatterndsl.patterndsl.MappingTree.Nodes.ParallelCalls.*;
import de.parallelpatterndsl.patterndsl.MappingTree.Nodes.Plain.*;
import de.parallelpatterndsl.patterndsl.abstractPatternTree.AbstractPatternTree;
import de.parallelpatterndsl.patterndsl.abstractPatternTree.DataElements.Data;
import de.parallelpatterndsl.patterndsl.abstractPatternTree.DataElements.TempData;
import de.parallelpatterndsl.patterndsl.abstractPatternTree.Nodes.FunctionNode;
import de.parallelpatterndsl.patterndsl.abstractPatternTree.Nodes.Functions.*;
import de.parallelpatterndsl.patterndsl.abstractPatternTree.Nodes.PatternNode;
import de.parallelpatterndsl.patterndsl.abstractPatternTree.Nodes.Plain.*;
import de.parallelpatterndsl.patterndsl.abstractPatternTree.Nodes.Plain.AdditionalArguments.MetaList;
import de.parallelpatterndsl.patterndsl.abstractPatternTree.Nodes.Plain.AdditionalArguments.MetaValue;
import de.parallelpatterndsl.patterndsl.hardwareDescription.Network;
import de.parallelpatterndsl.patterndsl.hardwareDescription.Processor;
import de.se_rwth.commons.logging.Log;
import org.json.simple.JSONArray;
import org.json.simple.JSONObject;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.HashSet;
import java.util.Optional;


/**
 * A class to simplify the reading on AMTs as a JSON file.
 */
public class JSONAMTInterpreter {


    private HashMap<String, FunctionMapping> functionTable;

    private AbstractPatternTree APT;

    private Network network;

    public AbstractMappingTree interpreteJSON(String path){
        JSONObject File = JSONAMTParser.readAMTJSON(path);

        for (int i = 0; i < ((JSONArray) File.get("Functions")).size(); i++) {
            JSONObject JSONNode = (JSONObject) ((JSONArray) File.get("Functions")).get(i);
            FunctionNode node = AbstractPatternTree.getFunctionTable().get(JSONNode.get("Name").toString());
            if (node.isAvailableAfterInlining()) {
                functionTable.put(node.getIdentifier(), generateAMT(JSONNode));
            }
        }

        AbstractMappingTree.setFunctionTable(functionTable);
        AbstractMappingTree AMT = new AbstractMappingTree((MainMapping) AbstractMappingTree.getFunctionTable().get("main"), APT.getGlobalVariableTable(), APT.getGlobalAssignments());
        return AMT;
    }

    public FunctionMapping generateAMT(JSONObject JSONNode) {

        FunctionNode node = AbstractPatternTree.getFunctionTable().get(JSONNode.get("Name").toString());
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

        ArrayList<MappingNode> children = new ArrayList<>();
        for (int i = 0; i < ((JSONArray) JSONNode.get("Children")).size(); i++) {
            JSONObject JSONAPTNode = (JSONObject) ((JSONArray) JSONNode.get("Children")).get(i);
            children.add(generateMappingNode(JSONAPTNode, functionMapping));
        }
        functionMapping.setChildren(children);

        return functionMapping;
    }

    private MappingNode generateMappingNode(JSONObject JSONNode, MappingNode parent) {
        if (JSONNode.get("Pattern Node Type").toString().equals("BranchCaseNode")) {
            return generateBranchCaseMapping(JSONNode, parent);
        } else if (JSONNode.get("Pattern Node Type").toString().equals("BranchNode")) {
            return generateBranchMapping( JSONNode, parent);
        } else if (JSONNode.get("Pattern Node Type").toString().equals("SerializedParallelCallNode")) {
            return generateSerializedParallelCallMapping( JSONNode, parent);
        } else if (JSONNode.get("Pattern Node Type").toString().equals("CallNode")) {
            return generateCallMapping( JSONNode, parent);
        } else if (JSONNode.get("Pattern Node Type").toString().equals("ComplexExpressionNode")) {
            return generateComplexExpressionMapping( JSONNode, parent);
        } else if (JSONNode.get("Pattern Node Type").toString().equals("ForEachLoopNode")) {
            return generateForEachLoopMapping( JSONNode, parent);
        } else if (JSONNode.get("Pattern Node Type").toString().equals("ForLoopNode")) {
            return generateForLoopMapping( JSONNode, parent);
        } else if (JSONNode.get("Pattern Node Type").toString().equals("ReturnNode")) {
            return generateReturnMapping( JSONNode, parent);
        } else if (JSONNode.get("Pattern Node Type").toString().equals("SimpleExpressionNode")) {
            return generateSimpleExpressionBlockMapping( JSONNode, parent);
        } else if (JSONNode.get("Pattern Node Type").toString().equals("WhileLoopNode")) {
            return generateWhileLoopMapping( JSONNode, parent);
        } else if (JSONNode.get("Pattern Node Type").toString().equals("LoopSkipNode")) {
            return generateLoopSkipMapping( JSONNode, parent);
        } else if (JSONNode.get("Pattern Node Type").toString().equals("JumpStatementNode")) {
            return generateJumpStatementMapping( JSONNode, parent);
        } else if (JSONNode.get("Pattern Node Type").toString().equals("JumpLabelNode")) {
            return generateJumpLabelMapping( JSONNode, parent);
        } else if (JSONNode.get("Pattern Node Type").toString().equals("FusedParallelCallNode")) {
            return generateFusedParallelCallMapping( JSONNode, parent);
        } else if (JSONNode.get("Pattern Node Type").toString().equals("GPUParallelCallNode")) {
            return generateGPUParallelCallMapping( JSONNode, parent);
        } else if (JSONNode.get("Pattern Node Type").toString().equals("ParallelCallNode")) {
            return generateParallelCallMapping( JSONNode, parent);
        } else if (JSONNode.get("Pattern Node Type").toString().equals("ReductionCallNode")) {
            return generateReductionCallMapping( JSONNode, parent);
        } else {
            Log.error("Pattern type not recognized");
            throw new RuntimeException("Critical error!");
        }
    }


    private MappingNode generateSerializedParallelCallMapping(JSONObject jsonNode, MappingNode parent) {

        ParallelCallNode node = (ParallelCallNode) NodeIDMapping.getMapping(jsonNode.get("Original ID").toString());
        ArrayList<Long> iterations = new ArrayList<>();
        ArrayList<Long> starts = new ArrayList<>();

        for (int i = 0; i < ((JSONArray) jsonNode.get("Starts")).size(); i++) {
            starts.add(Long.parseLong(((JSONArray) jsonNode.get("Starts")).get(i).toString()));
        }
        for (int i = 0; i < ((JSONArray) jsonNode.get("NumIterations")).size(); i++) {
            iterations.add(Long.parseLong(((JSONArray) jsonNode.get("NumIterations")).get(i).toString()));
        }

        SerializedParallelCallMapping result = new SerializedParallelCallMapping(Optional.of(parent), node.getVariableTable(), node, starts, iterations, null, 0);
        result.setChildren(new ArrayList<>());
        result.setDefinition(generateComplexExpressionMapping((JSONObject) ((JSONArray) jsonNode.get("Children")).get(0), result));

        return result;
    }


    private MappingNode generateFusedParallelCallMapping(JSONObject jsonNode, MappingNode parent) {
        ArrayList<MappingNode> children = new ArrayList<>();
        for (int i = 0; i < ((JSONArray) jsonNode.get("Children")).size(); i++) {
            JSONObject child = (JSONObject) ((JSONArray) jsonNode.get("Children")).get(i);
            children.add(generateMappingNode(child, parent));
        }

        FusedParallelCallMapping mapping = new FusedParallelCallMapping(Optional.of(parent), children.get(0).getVariableTable());

        mapping.setChildren(children);
        return mapping;
    }


    private MappingNode generateGPUParallelCallMapping(JSONObject jsonNode, MappingNode parent) {
        ParallelCallNode node = (ParallelCallNode) NodeIDMapping.getMapping(jsonNode.get("Original ID").toString());
        ArrayList<Long> iterations = new ArrayList<>();
        ArrayList<Long> starts = new ArrayList<>();

        for (int i = 0; i < ((JSONArray) jsonNode.get("Starts")).size(); i++) {
            starts.add(Long.parseLong(((JSONArray) jsonNode.get("Starts")).get(i).toString()));
        }
        for (int i = 0; i < ((JSONArray) jsonNode.get("NumIterations")).size(); i++) {
            iterations.add(Long.parseLong(((JSONArray) jsonNode.get("NumIterations")).get(i).toString()));
        }

        Processor processor = getProcessorFromPath(jsonNode.get("Cache Group").toString());

        int numThreads = Integer.parseInt(jsonNode.get("ThreadsPerBlock").toString());
        int numBlocks = Integer.parseInt(jsonNode.get("NumBlocks").toString());

        GPUParallelCallMapping result = new GPUParallelCallMapping(Optional.of(parent), node.getVariableTable(), node, starts, iterations, processor, numThreads, Optional.empty(), new HashSet<>(), numBlocks);
        result.setChildren(new ArrayList<>());
        result.setDefinition(generateComplexExpressionMapping((JSONObject) ((JSONArray) jsonNode.get("Children")).get(0), result));

        return result;
    }


    private MappingNode generateParallelCallMapping(JSONObject jsonNode, MappingNode parent) {
        ParallelCallNode node = (ParallelCallNode) NodeIDMapping.getMapping(jsonNode.get("Original ID").toString());
        ArrayList<Long> iterations = new ArrayList<>();
        ArrayList<Long> starts = new ArrayList<>();

        for (int i = 0; i < ((JSONArray) jsonNode.get("Starts")).size(); i++) {
            starts.add(Long.parseLong(((JSONArray) jsonNode.get("Starts")).get(i).toString()));
        }
        for (int i = 0; i < ((JSONArray) jsonNode.get("NumIterations")).size(); i++) {
            iterations.add(Long.parseLong(((JSONArray) jsonNode.get("NumIterations")).get(i).toString()));
        }

        Processor processor = getProcessorFromPath(jsonNode.get("Cache Group").toString());

        ParallelCallMapping result = new ParallelCallMapping(Optional.of(parent), node.getVariableTable(), node, starts, iterations, processor, processor.getCores(), Optional.empty(), new HashSet<>());
        result.setChildren(new ArrayList<>());
        result.setDefinition(generateComplexExpressionMapping((JSONObject) ((JSONArray) jsonNode.get("Children")).get(0), result));

        return result;
    }


    private MappingNode generateReductionCallMapping(JSONObject jsonNode, MappingNode parent) {

        ParallelCallNode node = (ParallelCallNode) NodeIDMapping.getMapping(jsonNode.get("Original ID").toString());
        ArrayList<Long> iterations = new ArrayList<>();
        ArrayList<Long> starts = new ArrayList<>();

        for (int i = 0; i < ((JSONArray) jsonNode.get("Starts")).size(); i++) {
            starts.add(Long.parseLong(((JSONArray) jsonNode.get("Starts")).get(i).toString()));
        }
        for (int i = 0; i < ((JSONArray) jsonNode.get("NumIterations")).size(); i++) {
            iterations.add(Long.parseLong(((JSONArray) jsonNode.get("NumIterations")).get(i).toString()));
        }

        Processor processor = getProcessorFromPath(jsonNode.get("Cache Group").toString());

        boolean isCombinerOnly = Boolean.parseBoolean(jsonNode.get("CombinerOnly").toString());

        int numThreads = Integer.parseInt(jsonNode.get("ThreadsPerBlock").toString());
        int numBlocks = Integer.parseInt(jsonNode.get("NumBlocks").toString());

        HashSet<TempData> inputTempData = new HashSet<>();
        HashSet<TempData> outputTempData = new HashSet<>();

        for (int i = 0; i < ((JSONArray) jsonNode.get("InputTempData")).size(); i++) {
            String tempName = ((JSONArray) jsonNode.get("InputTempData")).get(i).toString() + "_" + jsonNode.get("Original ID").toString();
            TempData tempData = new TempData(node.getCallExpression().getTypeName(),tempName);
            parent.getVariableTable().put(tempName, tempData);
            inputTempData.add(tempData);
        }

        for (int i = 0; i < ((JSONArray) jsonNode.get("OutputTempData")).size(); i++) {
            String tempName = ((JSONArray) jsonNode.get("OutputTempData")).get(i).toString() + "_" + jsonNode.get("Original ID").toString();
            TempData tempData = new TempData(node.getCallExpression().getTypeName(),tempName);
            parent.getVariableTable().put(tempName, tempData);
            outputTempData.add(tempData);
        }

        boolean onGPU = false;
        if (processor.getParent().getType().equals("GPU")) {
            onGPU = true;
        }

        ReductionCallMapping result = new ReductionCallMapping(Optional.of(parent), node.getVariableTable(), node, starts, iterations, processor, numThreads,isCombinerOnly, inputTempData, outputTempData, numBlocks, onGPU);
        result.setChildren(new ArrayList<>());
        result.setDefinition(generateComplexExpressionMapping((JSONObject) ((JSONArray) jsonNode.get("Children")).get(0), result));

        return result;
    }


    private Processor getProcessorFromPath(String cache_group) {

        String index[] = cache_group.split(":");

        return network.getNodes().get(Integer.parseInt(index[0])).getDevices().get(Integer.parseInt(index[1])).getProcessor().get(Integer.parseInt(index[2]));
    }


    private JumpStatementMapping generateJumpStatementMapping(JSONObject jsonNode, MappingNode parent) {
        JumpStatementNode node = (JumpStatementNode) NodeIDMapping.getMapping(jsonNode.get("Original ID").toString());
        JumpStatementMapping result = new JumpStatementMapping(Optional.of(parent), node.getVariableTable(), node, node.getClosingVars(), node.getLabel(), node.getOutputData(), AbstractMappingTree.getDefaultDevice().getParent());
        ComplexExpressionMapping resultExpression = generateComplexExpressionMapping((JSONObject) jsonNode.get("Result"), result);
        result.setResultExpression(resultExpression);
        return result;
    }

    private JumpLabelMapping generateJumpLabelMapping(JSONObject jsonNode, MappingNode parent) {
        JumpLabelNode node = (JumpLabelNode) NodeIDMapping.getMapping(jsonNode.get("Original ID").toString());
        JumpLabelMapping result = new JumpLabelMapping(Optional.of(parent), node.getVariableTable(), node, node.getLabel(), AbstractMappingTree.getDefaultDevice().getParent());
        return result;
    }


    private LoopSkipMapping generateLoopSkipMapping(JSONObject jsonNode, MappingNode parent) {
        LoopSkipNode node = (LoopSkipNode) NodeIDMapping.getMapping(jsonNode.get("Original ID").toString());
        return new LoopSkipMapping(Optional.of(parent), node.getVariableTable(), node, AbstractMappingTree.getDefaultDevice().getParent());
    }

    private ComplexExpressionMapping generateComplexExpressionMapping(JSONObject jsonNode, MappingNode parent) {
        ComplexExpressionNode node = (ComplexExpressionNode) NodeIDMapping.getMapping(jsonNode.get("Original ID").toString());
        ComplexExpressionMapping mapping = new ComplexExpressionMapping(Optional.of(parent), node.getVariableTable(), node, AbstractMappingTree.getDefaultDevice().getParent());
        ArrayList<MappingNode> children = new ArrayList<>();
        for (int i = 0; i < ((JSONArray) jsonNode.get("Children")).size(); i++) {
            JSONObject child = (JSONObject) ((JSONArray) jsonNode.get("Children")).get(i);
            children.add(generateMappingNode(child, parent));
        }
        mapping.setChildren(children);
        return mapping;
    }


    private CallMapping generateCallMapping(JSONObject jsonNode, MappingNode parent) {
        CallNode node = (CallNode) NodeIDMapping.getMapping(jsonNode.get("Original ID").toString());
        return new CallMapping(Optional.of(parent), node.getVariableTable(), node, AbstractMappingTree.getDefaultDevice().getParent());
    }

    private SimpleExpressionBlockMapping generateSimpleExpressionBlockMapping(JSONObject jsonNode, MappingNode parent) {
        SimpleExpressionBlockNode node = (SimpleExpressionBlockNode) NodeIDMapping.getMapping(jsonNode.get("Original ID").toString());
        return new SimpleExpressionBlockMapping(Optional.of(parent), node.getVariableTable(), node, AbstractMappingTree.getDefaultDevice().getParent());
    }

    private WhileLoopMapping generateWhileLoopMapping(JSONObject jsonNode, MappingNode parent) {
        WhileLoopNode node = (WhileLoopNode) NodeIDMapping.getMapping(jsonNode.get("Original ID").toString());
        WhileLoopMapping mapping = new WhileLoopMapping(Optional.of(parent), node.getVariableTable(), node, AbstractMappingTree.getDefaultDevice().getParent());
        ArrayList<MappingNode> children = new ArrayList<>();
        for (int i = 1; i < ((JSONArray) jsonNode.get("Children")).size(); i++) {
            JSONObject child = (JSONObject) ((JSONArray) jsonNode.get("Children")).get(i);
            children.add(generateMappingNode(child, mapping));
        }
        mapping.setChildren(children);
        mapping.setCondition(generateComplexExpressionMapping((JSONObject) ((JSONArray) jsonNode.get("Children")).get(0), mapping));
        return mapping;
    }

    private ReturnMapping generateReturnMapping(JSONObject jsonNode, MappingNode parent) {
        ReturnNode node = (ReturnNode) NodeIDMapping.getMapping(jsonNode.get("Original ID").toString());
        ReturnMapping mapping = new ReturnMapping(Optional.of(parent), node.getVariableTable(), node, AbstractMappingTree.getDefaultDevice().getParent());
        if (node.getChildren().size() == 1) {
            mapping.setResult(generateComplexExpressionMapping((JSONObject) jsonNode.get("Result"), mapping));
        }
        mapping.setChildren(new ArrayList<>());
        return mapping;
    }

    private ForLoopMapping generateForLoopMapping(JSONObject jsonNode, MappingNode parent) {
        ForLoopNode node = (ForLoopNode) NodeIDMapping.getMapping(jsonNode.get("Original ID").toString());
        ForLoopMapping mapping = new ForLoopMapping(Optional.of(parent), node.getVariableTable(), node, AbstractMappingTree.getDefaultDevice().getParent());
        mapping.setInitExpression(generateComplexExpressionMapping((JSONObject) jsonNode.get("Init"), mapping));
        mapping.setControlExpression(generateComplexExpressionMapping((JSONObject) jsonNode.get("Condition"), mapping));
        mapping.setUpdateExpression(generateComplexExpressionMapping((JSONObject) jsonNode.get("Update"), mapping));

        ArrayList<MappingNode> children = new ArrayList<>();
        for (int i = 3; i < ((JSONArray) jsonNode.get("Children")).size(); i++) {
            JSONObject child = (JSONObject) ((JSONArray) jsonNode.get("Children")).get(i);
            children.add(generateMappingNode(child, mapping));
        }
        mapping.setChildren(children);

        return mapping;
    }


    private ForEachLoopMapping generateForEachLoopMapping(JSONObject jsonNode, MappingNode parent) {
        ForEachLoopNode node = (ForEachLoopNode) NodeIDMapping.getMapping(jsonNode.get("Original ID").toString());
        ForEachLoopMapping mapping = new ForEachLoopMapping(Optional.of(parent), node.getVariableTable(), node, AbstractMappingTree.getDefaultDevice().getParent());

        mapping.setParsedList(generateComplexExpressionMapping((JSONObject) jsonNode.get("Condition"), mapping));

        ArrayList<MappingNode> children = new ArrayList<>();
        for (int i = 1; i < ((JSONArray) jsonNode.get("Children")).size(); i++) {
            JSONObject child = (JSONObject) ((JSONArray) jsonNode.get("Children")).get(i);
            children.add(generateMappingNode(child, mapping));
        }

        mapping.setChildren(children);
        return mapping;
    }

    private BranchMapping generateBranchMapping(JSONObject jsonNode, MappingNode parent) {
        BranchNode node = (BranchNode) NodeIDMapping.getMapping(jsonNode.get("Original ID").toString());
        BranchMapping mapping = new BranchMapping(Optional.of(parent), node.getVariableTable(), node, AbstractMappingTree.getDefaultDevice().getParent());
        ArrayList<MappingNode> children = new ArrayList<>();
        for (int i = 0; i < ((JSONArray) jsonNode.get("Children")).size(); i++) {
            JSONObject child = (JSONObject) ((JSONArray) jsonNode.get("Children")).get(i);
            children.add(generateMappingNode(child, mapping));
        }
        mapping.setChildren(children);
        return mapping;
    }

    private BranchCaseMapping generateBranchCaseMapping(JSONObject jsonNode, MappingNode parent) {
        BranchCaseNode node = (BranchCaseNode) NodeIDMapping.getMapping(jsonNode.get("Original ID").toString());
        BranchCaseMapping mapping = new BranchCaseMapping(Optional.of(parent), node.getVariableTable(), node, AbstractMappingTree.getDefaultDevice().getParent());
        ArrayList<MappingNode> children = new ArrayList<>();
        if (node.isHasCondition()) {
            mapping.setCondition(Optional.of(generateComplexExpressionMapping((JSONObject) jsonNode.get("Condition"), mapping)));
            for (int i = 1; i < ((JSONArray) jsonNode.get("Children")).size(); i++) {
                JSONObject child = (JSONObject) ((JSONArray) jsonNode.get("Children")).get(i);
                children.add(generateMappingNode(child, mapping));
            }
        } else {
            mapping.setCondition(Optional.empty());
            for (int i = 0; i < ((JSONArray) jsonNode.get("Children")).size(); i++) {
                JSONObject child = (JSONObject) ((JSONArray) jsonNode.get("Children")).get(i);
                children.add(generateMappingNode(child, parent));
            }
        }
        mapping.setChildren(children);

        return mapping;
    }

}
