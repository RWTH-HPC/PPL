package de.parallelpatterndsl.patterndsl.Postprocessing;

import de.parallelpatterndsl.patterndsl.abstractPatternTree.AbstractPatternTree;
import de.parallelpatterndsl.patterndsl.abstractPatternTree.DataElements.Data;
import de.parallelpatterndsl.patterndsl.abstractPatternTree.DataElements.DataAccess.DataAccess;
import de.parallelpatterndsl.patterndsl.abstractPatternTree.Nodes.FunctionNode;
import de.parallelpatterndsl.patterndsl.abstractPatternTree.Nodes.Functions.MainNode;
import de.parallelpatterndsl.patterndsl.abstractPatternTree.Nodes.PatternNode;
import de.parallelpatterndsl.patterndsl.abstractPatternTree.Nodes.Plain.ComplexExpressionNode;
import de.parallelpatterndsl.patterndsl.abstractPatternTree.Nodes.Plain.LoopSkipNode;
import de.parallelpatterndsl.patterndsl.abstractPatternTree.Nodes.Plain.ParallelCallNode;
import de.parallelpatterndsl.patterndsl.abstractPatternTree.Nodes.Plain.SimpleExpressionBlockNode;
import de.se_rwth.commons.logging.Log;

import java.util.ArrayList;

public class APTParentAwareDataTraces {

    public void generate() {

        MainNode mainNode = (MainNode) AbstractPatternTree.getFunctionTable().get("main");

        //generate data traces
        APTDataTraceGenerator traceGenerator = new APTDataTraceGenerator();
        traceGenerator.generateTraces(mainNode);

        Log.info("data trace generation finished!", "");


        for (FunctionNode node: AbstractPatternTree.getFunctionTable().values()) {
            generateParentAwareDataAccesses(node);
        }

        Log.info("parent aware data trace generation finished!", "");
    }


    private void generateParentAwareDataAccesses(PatternNode node) {
        if (node instanceof SimpleExpressionBlockNode || node instanceof ComplexExpressionNode || node instanceof LoopSkipNode) {
            return;
        } else {
            ArrayList<Data> inputData = node.getInputElements();
            ArrayList<Data> outputData = node.getOutputElements();
            ArrayList<DataAccess> inputAccesses = node.getInputAccesses();
            ArrayList<DataAccess> outputAccesses = node.getOutputAccesses();

            int numChildren = node.getChildren().size();
            if (node instanceof ParallelCallNode) {
                numChildren = 1;
            }

            for (int i = 0; i < numChildren; i++) {

                PatternNode childnode = node.getChildren().get(i);
                generateParentAwareDataAccesses(childnode);

                // Update input data
                for (Data element : childnode.getInputElements() ) {
                    if (node.getVariableTable().containsValue(element) && !inputData.contains(element)) {
                        if (!(node instanceof FunctionNode)) {
                            if (node.getParent().getVariableTable().containsValue(element)) {
                                inputData.add(element);
                            }
                        } else {
                            if (!element.getIdentifier().startsWith("INDEX")) {
                                inputData.add(element);
                            }
                        }
                    }
                }
                node.setInputElements(inputData);

                // Update output data
                for (Data element : childnode.getOutputElements() ) {
                    if (node.getVariableTable().containsValue(element) && !outputData.contains(element)) {
                        if (!(node instanceof FunctionNode)) {
                            if (node.getParent().getVariableTable().containsValue(element)) {
                                outputData.add(element);
                            }
                        } else {
                            outputData.add(element);
                        }
                    }
                }
                node.setOutputElements(outputData);

                // Update input data accesses
                for (DataAccess element : childnode.getInputAccesses() ) {
                    if (node.getVariableTable().containsValue(element.getData())) {
                        if (!(node instanceof FunctionNode)) {
                            if (node.getParent().getVariableTable().containsValue(element.getData())) {
                                inputAccesses.add(element);
                            }
                        } else {
                            inputAccesses.add(element);
                        }
                    }
                }
                node.setInputAccesses(inputAccesses);

                // Update output data accesses
                for (DataAccess element : childnode.getOutputAccesses() ) {
                    if (node.getVariableTable().containsValue(element.getData())) {
                        if (!(node instanceof FunctionNode)) {
                            if (node.getParent().getVariableTable().containsValue(element.getData())) {
                                outputAccesses.add(element);
                            }
                        } else {
                            outputAccesses.add(element);
                        }
                    }
                }
                node.setOutputAccesses(outputAccesses);
            }
        }
    }

}
