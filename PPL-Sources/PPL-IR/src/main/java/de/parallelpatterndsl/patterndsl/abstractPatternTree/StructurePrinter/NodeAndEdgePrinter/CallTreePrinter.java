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
 * Helper class that creates the call tree APT depiction.
 */
public class CallTreePrinter implements APTVisitor {

    private HashMap<String, FunctionNode> functionTable;

    private HashMap<PatternNode, String> nodeHashes = new HashMap<>();

    private ArrayList<PatternNode> currentParents = new ArrayList<>();

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
    private void edge2ParentFunction(CallNode node) {
        arcBuilder.append("dot.edge('");
        arcBuilder.append(nodeHashes.get(currentParents.get(currentParents.size() - 1)));
        arcBuilder.append("', '");
        arcBuilder.append(nodeHashes.get(node));
        arcBuilder.append("')\n");
    }

    @Override
    public void visit(SerialNode node) {

        nodeBuilder.append("dot.node('");
        nodeBuilder.append(nodeHashes.get(node));
        nodeBuilder.append("', 'Main')\n");

    }

    @Override
    public void endVisit(SerialNode node) {
        nodeHashes.remove(node);
        currentParents.remove(currentParents.size() - 1);
    }

    @Override
    public void visit(RecursionNode node) {

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
    }

    @Override
    public void visit(CallNode node) {
        nodeHashes.put(node,RandomStringGenerator.getAlphaNumericString());
        nodeBuilder.append("dot.node('");
        nodeBuilder.append(nodeHashes.get(node));
        nodeBuilder.append("', 'Call: ");
        nodeBuilder.append(node.getFunctionIdentifier());
        nodeBuilder.append("', style=\"filled\", fillcolor=\"green\")\n");

        edge2ParentFunction(node);

        currentParents.add(node);

    }

    @Override
    public void endVisit(CallNode node) {
        nodeHashes.remove(node);
        currentParents.remove(currentParents.size() - 1);
    }

    @Override
    public void visit(ParallelCallNode node) {
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
        nodeBuilder.append("', style=\"filled\", fillcolor=\"red\")\n");

        edge2ParentFunction(node);

        currentParents.add(node);
    }

    @Override
    public void endVisit(ParallelCallNode node) {
        nodeHashes.remove(node);
        currentParents.remove(currentParents.size() - 1);
    }
    /***************************************
     *
     *
     * Visitor necessities.
     *
     *
     ****************************************/
    private CallTreePrinter realThis = this;

    @Override
    public CallTreePrinter getRealThis() {
        return realThis;
    }

    public void setRealThis(CallTreePrinter realThis) {
        this.realThis = realThis;
    }

}
