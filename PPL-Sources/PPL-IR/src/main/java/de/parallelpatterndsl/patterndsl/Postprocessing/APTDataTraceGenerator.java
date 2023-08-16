package de.parallelpatterndsl.patterndsl.Postprocessing;

import de.parallelpatterndsl.patterndsl.PatternTypes;
import de.parallelpatterndsl.patterndsl.abstractPatternTree.AbstractPatternTree;
import de.parallelpatterndsl.patterndsl.abstractPatternTree.DataElements.Data;
import de.parallelpatterndsl.patterndsl.abstractPatternTree.DataElements.DataAccess.DataAccess;
import de.parallelpatterndsl.patterndsl.abstractPatternTree.DataElements.FunctionInlineData;
import de.parallelpatterndsl.patterndsl.abstractPatternTree.Nodes.FunctionNode;
import de.parallelpatterndsl.patterndsl.abstractPatternTree.Nodes.Functions.*;
import de.parallelpatterndsl.patterndsl.abstractPatternTree.Nodes.PatternNode;
import de.parallelpatterndsl.patterndsl.abstractPatternTree.Nodes.Plain.*;
import de.parallelpatterndsl.patterndsl.abstractPatternTree.Visitor.APTVisitor;
import de.parallelpatterndsl.patterndsl.expressions.IRLExpression;

import java.util.ArrayList;

/**
 * Class that traverses the APT and generates the data traces for each scope.
 */
public class APTDataTraceGenerator implements APTVisitor {

    /**
     * List of currently active parallel patterns.
     */
    private ArrayList<PatternTypes> patternNesting = new ArrayList<>();

    public void generateTraces(MainNode mainNode) {
        mainNode.accept(getRealThis());
    }


    @Override
    public void visit(ParallelCallNode node) {
        FunctionNode function = AbstractPatternTree.getFunctionTable().get(node.getFunctionIdentifier());
        if (function instanceof MapNode) {
            patternNesting.add(PatternTypes.MAP);
        } else if (function instanceof StencilNode) {
            patternNesting.add(PatternTypes.STENCIL);
        } else if (function instanceof ReduceNode) {
            patternNesting.add(PatternTypes.REDUCE);
        } else if (function instanceof RecursionNode) {
            patternNesting.add(PatternTypes.RECURSION);
        } else if (function instanceof DynamicProgrammingNode) {
            patternNesting.add(PatternTypes.DYNAMIC_PROGRAMMING);
        }
    }

    @Override
    public void endVisit(ParallelCallNode node) {
        patternNesting.remove(patternNesting.size() - 1);
    }

    @Override
    public void visit(CallNode node) {
        patternNesting.add(PatternTypes.SEQUENTIAL);
    }

    @Override
    public void endVisit(CallNode node) {
        patternNesting.remove(patternNesting.size() - 1);
    }

    @Override
    public void visit(SerialNode node) {
        patternNesting.add(PatternTypes.SEQUENTIAL);
    }

    @Override
    public void endVisit(SerialNode node) {
        patternNesting.remove(patternNesting.size() - 1);
    }


    @Override
    public void visit(LoopSkipNode node) {
        ArrayList<DataAccess> accesses = new ArrayList<>();
        for (Data data : node.getVariableTable().values() ) {
            accesses.add(new DataAccess(data, false));
            //accesses.add(new DataAccess(data, true));
        }

        ArrayList<DataAccess> inputAccesses = new ArrayList<>();
        ArrayList<DataAccess> outputAccesses = new ArrayList<>();

        ArrayList<Data> inputData = new ArrayList<>();
        ArrayList<Data> outputData = new ArrayList<>();

        // Create the data traces based on the given data accesses.
        for (DataAccess dataAccess : accesses) {
            if (dataAccess.getData() instanceof FunctionInlineData) {
                continue;
            }
            dataAccess.getData().getTrace().addTraceElement(node, dataAccess);
            if (dataAccess.isReadAccess()) {
                inputAccesses.add(dataAccess);
                if (!inputData.contains(dataAccess.getData())) {
                    inputData.add(dataAccess.getData());
                }
            } else {
                outputAccesses.add(dataAccess);
                if (!outputData.contains(dataAccess.getData())) {
                    outputData.add(dataAccess.getData());
                }
            }
        }

        // Set the in-/output accesses and elements for node.
        node.setInputAccesses(inputAccesses);
        node.setInputElements(inputData);
        node.setOutputAccesses(outputAccesses);
        node.setOutputElements(outputData);
    }

    @Override
    public void visit(ComplexExpressionNode node) {
        // Get the data accesses from the node
        IRLExpression expression = node.getExpression();
        ArrayList<DataAccess> accesses = expression.getDataAccesses(patternNesting.get(patternNesting.size() - 1));

        if (expression.hasProfilingInfo() || expression.hasExit()) {
            for (Data data : node.getVariableTable().values() ) {
                //accesses.add(new DataAccess(data, false));
                accesses.add(new DataAccess(data, true));
            }
        }

        // Generate the correct data access for pattern calls
        if (node.getParent() instanceof ParallelCallNode) {
            accesses = expression.getDataAccesses(patternNesting.get(patternNesting.size() - 2));
        }

        ArrayList<DataAccess> inputAccesses = new ArrayList<>();
        ArrayList<DataAccess> outputAccesses = new ArrayList<>();

        ArrayList<Data> inputData = new ArrayList<>();
        ArrayList<Data> outputData = new ArrayList<>();

        // Create the data traces based on the given data accesses.
        for (DataAccess dataAccess : accesses) {
            if (dataAccess.getData() instanceof FunctionInlineData) {
                continue;
            }
            dataAccess.getData().getTrace().addTraceElement(node, dataAccess);
            if (dataAccess.isReadAccess()) {
                inputAccesses.add(dataAccess);
                if (!inputData.contains(dataAccess.getData())) {
                    inputData.add(dataAccess.getData());
                }
            } else {
                outputAccesses.add(dataAccess);
                if (!outputData.contains(dataAccess.getData())) {
                    outputData.add(dataAccess.getData());
                }
            }
        }

        // Set the in-/output accesses and elements for node.
        node.setInputAccesses(inputAccesses);
        node.setInputElements(inputData);
        node.setOutputAccesses(outputAccesses);
        node.setOutputElements(outputData);

    }

    @Override
    public void visit(SimpleExpressionBlockNode node) {
        ArrayList<DataAccess> inputAccesses = new ArrayList<>();
        ArrayList<DataAccess> outputAccesses = new ArrayList<>();

        ArrayList<Data> inputData = new ArrayList<>();
        ArrayList<Data> outputData = new ArrayList<>();
        for (IRLExpression expression : node.getExpressionList()) {
            // Get the data accesses from the node
            ArrayList<DataAccess> accesses = expression.getDataAccesses(patternNesting.get(patternNesting.size() - 1));


            // Create the data traces based on the given data accesses.
            for (DataAccess dataAccess : accesses) {
                if (dataAccess.getData() instanceof FunctionInlineData) {
                    continue;
                }
                dataAccess.getData().getTrace().addTraceElement(node, dataAccess);
                if (dataAccess.isReadAccess()) {
                    inputAccesses.add(dataAccess);
                    if (!inputData.contains(dataAccess.getData())) {
                        inputData.add(dataAccess.getData());
                    }
                } else {
                    outputAccesses.add(dataAccess);
                    if (!outputData.contains(dataAccess.getData())) {
                        outputData.add(dataAccess.getData());
                    }
                }
            }

            // Set the in-/output accesses and elements for node.
            node.setInputAccesses(inputAccesses);
            node.setInputElements(inputData);
            node.setOutputAccesses(outputAccesses);
            node.setOutputElements(outputData);
        }
    }


    /**
     * Visitor support functions.
     */
    private APTVisitor realThis = this;

    @Override
    public APTVisitor getRealThis() {
        return realThis;
    }

    @Override
    public void setRealThis(APTVisitor realThis) {
        this.realThis = realThis;
    }


}
