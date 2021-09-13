package de.parallelpatterndsl.patterndsl.abstractPatternTree.StructurePrinter.NodeAndEdgePrinter;

import de.parallelpatterndsl.patterndsl.abstractPatternTree.AbstractPatternTree;
import de.parallelpatterndsl.patterndsl.abstractPatternTree.Nodes.FunctionNode;
import de.parallelpatterndsl.patterndsl.abstractPatternTree.Nodes.Functions.*;
import de.parallelpatterndsl.patterndsl.abstractPatternTree.Nodes.PatternNode;
import de.parallelpatterndsl.patterndsl.abstractPatternTree.Nodes.Plain.*;
import de.parallelpatterndsl.patterndsl.abstractPatternTree.Visitor.APTVisitor;
import de.parallelpatterndsl.patterndsl.helperLibrary.RandomStringGenerator;

import java.util.ArrayList;
import java.util.HashMap;

/**
 * Helper class that creates the pattern nesting APT depiction.
 */
public class PatternNestingTreePrinter implements APTVisitor {

    private HashMap<String, FunctionNode> functionTable;

    private ArrayList<Boolean> currentNodeIsSerial = new ArrayList<>();

    private PatternNode currentSerialNode;

    private ArrayList<PatternNode> addedSerial = new ArrayList<>();

    private String currentSerialHash;

    private HashMap<PatternNode, String> nodeHashes = new HashMap<>();

    private ArrayList<PatternNode> currentParents = new ArrayList<>();

    private StringBuilder nodeBuilder = new StringBuilder();

    private StringBuilder arcBuilder = new StringBuilder();

    private testParallelDesc descTester = new testParallelDesc();

    /**
     *
     * @param mainNode
     * @return The nodes and edges for the complete APT.
     */
    public String generate(MainNode mainNode) {

        functionTable = AbstractPatternTree.getFunctionTable();
        RandomStringGenerator.setN(20);
        nodeHashes.put(mainNode,RandomStringGenerator.getAlphaNumericString());
        currentParents.add(mainNode);
        mainNode.accept(this.getRealThis());


        //handle recursive definitions
        for (FunctionNode node : functionTable.values() ) {
            if (node instanceof RecursionNode) {
                currentParents.add(node);
                nodeHashes.put(node,RandomStringGenerator.getAlphaNumericString());
                node.accept(this.getRealThis());
            }
        }

        RandomStringGenerator.setN(10);

        return nodeBuilder.append(arcBuilder).toString();
    }

    /**
     * generates the edge between the current node and its ancestor.
     * @param node
     */
    private void edge2Parent(PatternNode node) {
        arcBuilder.append("dot.edge('");
        arcBuilder.append(nodeHashes.get(currentParents.get(currentParents.size() - 1)));
        arcBuilder.append("', '");
        arcBuilder.append(nodeHashes.get(node));
        arcBuilder.append("')\n");
    }

    @Override
    public void visit(SerialNode node) {
        currentNodeIsSerial.add(false);
        nodeBuilder.append("dot.node('");
        nodeBuilder.append(nodeHashes.get(node));
        nodeBuilder.append("', 'Main')\n");

    }

    @Override
    public void endVisit(SerialNode node) {
        nodeHashes.remove(node);
        currentParents.remove(currentParents.size() - 1);
        currentNodeIsSerial.clear();
    }

    @Override
    public void visit(RecursionNode node) {
        currentNodeIsSerial.add(false);
        nodeBuilder.append("dot.node('");
        nodeBuilder.append(nodeHashes.get(node));
        nodeBuilder.append("', 'Recursion: ");
        nodeBuilder.append(node.getIdentifier());
        nodeBuilder.append("')\n");

    }

    @Override
    public void endVisit(RecursionNode node) {
        nodeHashes.remove(node);
        currentParents.remove(currentParents.size() - 1);
        currentNodeIsSerial.clear();
    }

    @Override
    public void visit(ComplexExpressionNode node) {
        if (!currentNodeIsSerial.get(currentNodeIsSerial.size() - 1)) {
            currentNodeIsSerial.add(true);
            currentSerialNode = node;
            nodeHashes.put(node, RandomStringGenerator.getAlphaNumericString());
            nodeBuilder.append("dot.node('");
            nodeBuilder.append(nodeHashes.get(node));
            nodeBuilder.append("', 'Serial', style=\"filled\", fillcolor=\"lawngreen\")\n");
            edge2Parent(node);
            currentParents.add(currentSerialNode);
            addedSerial.add(node);
        }


    }

    @Override
    public void endVisit(ComplexExpressionNode node) {
        if (addedSerial.get(addedSerial.size()-1) == node) {
            currentParents.remove(currentParents.size() - 1);
        }
    }


    @Override
    public void visit(SimpleExpressionBlockNode node) {
        if (!currentNodeIsSerial.get(currentNodeIsSerial.size() - 1)) {
            currentNodeIsSerial.add(true);
            currentSerialNode = node;
            nodeHashes.put(node, RandomStringGenerator.getAlphaNumericString());
            nodeBuilder.append("dot.node('");
            nodeBuilder.append(nodeHashes.get(node));
            nodeBuilder.append("', 'Serial', style=\"filled\", fillcolor=\"lawngreen\")\n");
            edge2Parent(node);
            currentParents.add(currentSerialNode);
            addedSerial.add(node);
        }
    }

    @Override
    public void endVisit(SimpleExpressionBlockNode node) {
        if (addedSerial.get(addedSerial.size()-1) == node) {
            currentParents.remove(currentParents.size() - 1);
        }
    }

    @Override
    public void visit(ForLoopNode node) {
        if (descTester.getParDesc(node)) {
            currentNodeIsSerial.add(true);
            currentSerialNode = node;
            nodeHashes.put(node, RandomStringGenerator.getAlphaNumericString());
            nodeBuilder.append("dot.node('");
            nodeBuilder.append(nodeHashes.get(node));
            nodeBuilder.append("', 'For Loop', style=\"filled\", fillcolor=\"cyan\")\n");
            edge2Parent(node);
            currentParents.add(currentSerialNode);
        }
    }

    @Override
    public void endVisit(ForLoopNode node) {
        if (descTester.getParDesc(node)) {
            currentParents.remove(currentParents.size() - 1);
        }
    }

    @Override
    public void visit(ForEachLoopNode node) {
        if (descTester.getParDesc(node)) {
            currentNodeIsSerial.add(true);
            currentSerialNode = node;
            nodeHashes.put(node, RandomStringGenerator.getAlphaNumericString());
            nodeBuilder.append("dot.node('");
            nodeBuilder.append(nodeHashes.get(node));
            nodeBuilder.append("', 'For Loop', style=\"filled\", fillcolor=\"cyan\")\n");
            edge2Parent(node);
            currentParents.add(currentSerialNode);
        }
    }

    @Override
    public void endVisit(ForEachLoopNode node) {
        if (descTester.getParDesc(node)) {
            currentParents.remove(currentParents.size() - 1);
        }
    }

    @Override
    public void visit(WhileLoopNode node) {
        if (descTester.getParDesc(node)) {
            currentNodeIsSerial.add(true);
            currentSerialNode = node;
            nodeHashes.put(node, RandomStringGenerator.getAlphaNumericString());
            nodeBuilder.append("dot.node('");
            nodeBuilder.append(nodeHashes.get(node));
            nodeBuilder.append("', 'While Loop', style=\"filled\", fillcolor=\"cyan\")\n");
            edge2Parent(node);
            currentParents.add(currentSerialNode);
        }
    }

    @Override
    public void endVisit(WhileLoopNode node) {
        if (descTester.getParDesc(node)) {
            currentParents.remove(currentParents.size() - 1);
        }
    }

    @Override
    public void visit(BranchNode node) {
        if (descTester.getParDesc(node)) {
            currentNodeIsSerial.add(true);
            currentSerialNode = node;
            nodeHashes.put(node, RandomStringGenerator.getAlphaNumericString());
            nodeBuilder.append("dot.node('");
            nodeBuilder.append(nodeHashes.get(node));
            nodeBuilder.append("', 'Branch', style=\"filled\", fillcolor=\"cyan\")\n");
            edge2Parent(node);
            currentParents.add(currentSerialNode);
        }
    }

    @Override
    public void endVisit(BranchNode node) {
        if (descTester.getParDesc(node)) {
            currentParents.remove(currentParents.size() - 1);
        }
    }

    @Override
    public void visit(ParallelCallNode node) {

        currentNodeIsSerial.add(false);

        nodeHashes.put(node,RandomStringGenerator.getAlphaNumericString());
        nodeBuilder.append("dot.node('");
        nodeBuilder.append(nodeHashes.get(node));

        FunctionNode function = functionTable.get(node.getFunctionIdentifier());
        if (function instanceof MapNode) {
            nodeBuilder.append("', 'Map: ");
        } else if (function instanceof ReduceNode){
            nodeBuilder.append("', 'Reduction: ");
        } else if (function instanceof StencilNode) {
            nodeBuilder.append("', 'Stencil: ");
        } else if (function instanceof RecursionNode) {
            nodeBuilder.append("', 'Recursion: ");
        } else if (function instanceof DynamicProgrammingNode) {
            nodeBuilder.append("', 'Dynamic Programming: ");
        }

        nodeBuilder.append(node.getFunctionIdentifier());
        nodeBuilder.append("', style=\"filled\", fillcolor=\"orangered\")\n");

        edge2Parent(node);

        currentParents.add(node);
    }

    @Override
    public void endVisit(ParallelCallNode node) {
        nodeHashes.remove(node);
        currentParents.remove(currentParents.size() - 1);
        currentNodeIsSerial.remove(currentNodeIsSerial.size() - 1);
    }
    /***************************************
     *
     *
     * Visitor necessities.
     *
     *
     ****************************************/
    private PatternNestingTreePrinter realThis = this;

    @Override
    public PatternNestingTreePrinter getRealThis() {
        return realThis;
    }

    public void setRealThis(PatternNestingTreePrinter realThis) {
        this.realThis = realThis;
    }


    private class testParallelDesc implements APTVisitor {
        private Boolean res = false;


        public Boolean getParDesc(PatternNode node) {
            res = false;

            node.accept(this.getRealThis());

            return res;
        }

        @Override
        public void visit(ParallelCallNode node) {

            res = true;
        }



        /***************************************
         *
         *
         * Visitor necessities.
         *
         *
         ****************************************/
        private testParallelDesc realThis = this;

        @Override
        public testParallelDesc getRealThis() {
            return realThis;
        }

        public void setRealThis(testParallelDesc realThis) {
            this.realThis = realThis;
        }
    }
}
