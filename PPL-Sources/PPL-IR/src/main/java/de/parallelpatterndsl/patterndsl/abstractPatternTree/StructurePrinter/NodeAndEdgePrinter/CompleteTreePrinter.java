package de.parallelpatterndsl.patterndsl.abstractPatternTree.StructurePrinter.NodeAndEdgePrinter;

import de.parallelpatterndsl.patterndsl.abstractPatternTree.AbstractPatternTree;
import de.parallelpatterndsl.patterndsl.abstractPatternTree.Nodes.FunctionNode;
import de.parallelpatterndsl.patterndsl.abstractPatternTree.Nodes.Functions.*;
import de.parallelpatterndsl.patterndsl.abstractPatternTree.Nodes.PatternNode;
import de.parallelpatterndsl.patterndsl.abstractPatternTree.Nodes.Plain.*;
import de.parallelpatterndsl.patterndsl.abstractPatternTree.Visitor.APTVisitor;
import de.parallelpatterndsl.patterndsl.helperLibrary.RandomStringGenerator;

import java.util.HashMap;


/**
 * Helper class that creates the complete APT depiction.
 */
public class CompleteTreePrinter implements APTVisitor {

    private HashMap<String, FunctionNode> functionTable;

    private HashMap<PatternNode, String> nodeHashes = new HashMap<>();

    private StringBuilder nodeBuilder = new StringBuilder();

    private StringBuilder arcBuilder = new StringBuilder();

    /**
     *
     * @param mainNode
     * @return The nodes and edges for the complete APT.
     */
    public String generate(MainNode mainNode) {
        functionTable = AbstractPatternTree.getFunctionTable();
        RandomStringGenerator.setN(20);
        nodeHashes.put(mainNode,RandomStringGenerator.getAlphaNumericString());
        mainNode.accept(this.getRealThis());


        //handle recursive definitions
        for (FunctionNode node : functionTable.values() ) {
            if (node instanceof RecursionNode) {
                nodeHashes.put(node,RandomStringGenerator.getAlphaNumericString());
                node.accept(this.getRealThis());
            }
        }

        RandomStringGenerator.setN(10);

        return nodeBuilder.append(arcBuilder).toString();
    }

    /**
     * Creates the edges to all child nodes depending on the current node.
     * @param node
     */
    private void createChildEdges(PatternNode node) {
        for (PatternNode child: node.getChildren()) {
            nodeHashes.put(child,RandomStringGenerator.getAlphaNumericString());
            arcBuilder.append("dot.edge('");
            arcBuilder.append(nodeHashes.get(node));
            arcBuilder.append("', '");
            arcBuilder.append(nodeHashes.get(child));
            arcBuilder.append("')\n");
        }
    }

    @Override
    public void visit(SerialNode node) {

        nodeBuilder.append("dot.node('");
        nodeBuilder.append(nodeHashes.get(node));
        nodeBuilder.append("', 'Main')\n");

        createChildEdges(node);
    }

    @Override
    public void endVisit(SerialNode node) {
        nodeHashes.remove(node);
    }

    @Override
    public void visit(RecursionNode node) {

        nodeBuilder.append("dot.node('");
        nodeBuilder.append(nodeHashes.get(node));
        nodeBuilder.append("', 'Recursion: ");
        nodeBuilder.append(node.getIdentifier());
        nodeBuilder.append("')\n");

        createChildEdges(node);
    }

    @Override
    public void endVisit(RecursionNode node) {
        nodeHashes.remove(node);
    }

    public void visit(BranchNode node) {
        nodeBuilder.append("dot.node('");
        nodeBuilder.append(nodeHashes.get(node));
        nodeBuilder.append("', 'Branch')\n");

        createChildEdges(node);
    }

    @Override
    public void endVisit(BranchNode node) {
        nodeHashes.remove(node);
    }

    @Override
    public void visit(BranchCaseNode node) {
        nodeBuilder.append("dot.node('");
        nodeBuilder.append(nodeHashes.get(node));
        nodeBuilder.append("', 'Case')\n");

        createChildEdges(node);
    }

    @Override
    public void endVisit(BranchCaseNode node) {
        nodeHashes.remove(node);
    }

    @Override
    public void visit(CallNode node) {
        nodeBuilder.append("dot.node('");
        nodeBuilder.append(nodeHashes.get(node));
        nodeBuilder.append("', 'Call: ");
        nodeBuilder.append(node.getFunctionIdentifier());
        nodeBuilder.append("', style=\"filled\", fillcolor=\"green\")\n");

        createChildEdges(node);
    }

    @Override
    public void endVisit(CallNode node) {
        nodeHashes.remove(node);
    }

    @Override
    public void visit(ComplexExpressionNode node) {
        nodeBuilder.append("dot.node('");
        nodeBuilder.append(nodeHashes.get(node));
        nodeBuilder.append("', 'Expression')\n");

        createChildEdges(node);
    }

    @Override
    public void endVisit(ComplexExpressionNode node) {
        nodeHashes.remove(node);
    }

    @Override
    public void visit(ForEachLoopNode node) {
        nodeBuilder.append("dot.node('");
        nodeBuilder.append(nodeHashes.get(node));
        nodeBuilder.append("', 'For-Each-Loop')\n");

        createChildEdges(node);
    }

    @Override
    public void endVisit(ForEachLoopNode node) {
        nodeHashes.remove(node);
    }

    @Override
    public void visit(ForLoopNode node) {
        nodeBuilder.append("dot.node('");
        nodeBuilder.append(nodeHashes.get(node));
        nodeBuilder.append("', 'For-Loop')\n");

        createChildEdges(node);
    }

    @Override
    public void endVisit(ForLoopNode node) {
        nodeHashes.remove(node);
    }

    @Override
    public void visit(WhileLoopNode node) {
        nodeBuilder.append("dot.node('");
        nodeBuilder.append(nodeHashes.get(node));
        nodeBuilder.append("', 'While-Loop')\n");

        createChildEdges(node);
    }

    @Override
    public void endVisit(WhileLoopNode node) {
        nodeHashes.remove(node);
    }

    @Override
    public void visit(ReturnNode node) {
        nodeBuilder.append("dot.node('");
        nodeBuilder.append(nodeHashes.get(node));
        nodeBuilder.append("', 'Return')\n");

        createChildEdges(node);
    }

    @Override
    public void endVisit(ReturnNode node) {
        nodeHashes.remove(node);
    }

    @Override
    public void visit(SimpleExpressionBlockNode node) {
        nodeBuilder.append("dot.node('");
        nodeBuilder.append(nodeHashes.get(node));
        nodeBuilder.append("', 'Simple Expression')\n");

        createChildEdges(node);
    }

    @Override
    public void endVisit(SimpleExpressionBlockNode node) {
        nodeHashes.remove(node);
    }

    @Override
    public void visit(ParallelCallNode node) {
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
        nodeBuilder.append("', style=\"filled\", fillcolor=\"red\")\n");
        createChildEdges(node);
    }

    @Override
    public void endVisit(ParallelCallNode node) {
        nodeHashes.remove(node);
    }
    /***************************************
     *
     *
     * Visitor necessities.
     *
     *
     ****************************************/
    private CompleteTreePrinter realThis = this;

    @Override
    public CompleteTreePrinter getRealThis() {
        return realThis;
    }

    public void setRealThis(CompleteTreePrinter realThis) {
        this.realThis = realThis;
    }

}
