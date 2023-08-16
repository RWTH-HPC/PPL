package de.parallelpatterndsl.patterndsl.Preprocessing;

import de.parallelpatterndsl.patterndsl.abstractPatternTree.AbstractPatternTree;
import de.parallelpatterndsl.patterndsl.abstractPatternTree.DataElements.ArrayData;
import de.parallelpatterndsl.patterndsl.abstractPatternTree.DataElements.Data;
import de.parallelpatterndsl.patterndsl.abstractPatternTree.DataElements.PrimitiveData;
import de.parallelpatterndsl.patterndsl.abstractPatternTree.Nodes.PatternNode;
import de.parallelpatterndsl.patterndsl.abstractPatternTree.Nodes.Plain.*;
import de.parallelpatterndsl.patterndsl.abstractPatternTree.Visitor.APTVisitor;
import de.parallelpatterndsl.patterndsl.expressions.IRLExpression;
import de.parallelpatterndsl.patterndsl.helperLibrary.DeepCopyHelper;
import de.parallelpatterndsl.patterndsl.helperLibrary.RandomStringGenerator;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.Map;
import java.util.stream.Collectors;

/**
 * This class defines the unrolling process for loop nodes.
 */
public class APTLoopUnrollHandler {

    private AbstractPatternTree APT;

    public APTLoopUnrollHandler(AbstractPatternTree APT) {
        this.APT = APT;
    }

    public boolean generate() {
        boolean result = false;
        ArrayList<PatternNode> newRootChildren = new ArrayList<>();
        for (PatternNode node: APT.getRoot().getChildren() ) {
            if (node instanceof ForLoopNode) {
                int tester = newRootChildren.size();
                newRootChildren.addAll(unrollForLoop((ForLoopNode) node));
                if (tester + 1 < newRootChildren.size()) {
                     result = true;
                }
            } else {
                newRootChildren.add(node);
            }
        }

        APT.getRoot().setChildren(newRootChildren);
        return result;
    }

    /**
     * Unrolls a loop if applicable and returns the set of nodes produced.
     * @param node
     * @return
     */
    public ArrayList<PatternNode> unrollForLoop(ForLoopNode node) {

        /*
        Currently, only simple loops (no break/continue, known length and no write accesses to loop control variable.)
        1. get length estimation
        2. Create Array to copy
        3. Add start value to result
        4. Copy running var and add to variable table
        5. Set copy array to children of node and replace the running var
        6. Add the update to the running clause
        7. Add copies to the result
         */

        String unrollIdentifier = RandomStringGenerator.getAlphaNumericString();

        ArrayList<PatternNode> result = new ArrayList<>();
        if (!node.isSimple() || !node.isHasParallelDescendants()) {
            result.add(node);
            return result;
        }

        int numIterations = node.getNumIterations();

        ArrayList<PatternNode> childrenToCopy = new ArrayList<>();

        // create iteration data
        Data iterationData;
        if (node.getLoopControlVariable() instanceof PrimitiveData) {
            iterationData = new PrimitiveData(node.getLoopControlVariable().getIdentifier() + "_" + unrollIdentifier, node.getLoopControlVariable().getTypeName(), false);
        } else {
            iterationData = new ArrayData(node.getLoopControlVariable().getIdentifier() + "_" + unrollIdentifier, node.getLoopControlVariable().getTypeName(), false, (ArrayList<Integer>) ((ArrayData) node.getLoopControlVariable()).getShape().clone(), false);
        }

        HashMap<String, Data> localTable = new HashMap<>(node.getVariableTable());
        for (Map.Entry<String, Data> entry: node.getParent().getVariableTable().entrySet() ) {
            if (!localTable.containsKey(entry.getKey())) {
                localTable.put(entry.getKey(), entry.getValue());
            }
        }
        localTable.put(iterationData.getIdentifier(), iterationData);


        //Find and replace local vars

        HashMap<Data, Data> localReplacements = new HashMap<>();

        for (Data data : node.getVariableTable().values().stream().filter(x -> x instanceof ArrayData || x instanceof PrimitiveData).filter(x -> !node.getParent().getVariableTable().containsValue(x)).collect(Collectors.toList()) ) {
            Data replacement;
            if ( data instanceof PrimitiveData) {
                replacement = new PrimitiveData(data.getIdentifier() + "_" + unrollIdentifier, data.getTypeName(), false);
            } else {
                replacement = new ArrayData(data.getIdentifier() + "_" + unrollIdentifier, data.getTypeName(), false, new ArrayList<>(((ArrayData) data).getShape()), false);
            }
            localReplacements.put(data, replacement);
            localTable.put(replacement.getIdentifier(), replacement);
        }


        DeepCopyHelper.addScope(localTable);
        // Replace new variables within child nodes
        for (int i = 3; i < node.getChildren().size(); i++) {
            // replace loop control var
            PatternNode currentNode = node.getChildren().get(i).deepCopy();
            FindAndReplaceLoopVariable findAndReplace = new FindAndReplaceLoopVariable(node.getLoopControlVariable(), localTable.get(iterationData.getIdentifier()), currentNode);
            findAndReplace.doReplace();
            // replace loop local vars
            for (Map.Entry<Data, Data> entry : localReplacements.entrySet() ) {
                FindAndReplaceLoopVariable findAndReplaceLocals = new FindAndReplaceLoopVariable(entry.getKey(), entry.getValue(), currentNode);
                findAndReplaceLocals.doReplace();
            }
            // create iteration template
            childrenToCopy.add(currentNode);
        }

        // create update clause for loop control var
        ComplexExpressionNode update = (ComplexExpressionNode) node.getChildren().get(2).deepCopy();
        update.getExpression().replaceDataElement(node.getLoopControlVariable(), localTable.get(iterationData.getIdentifier()));
        childrenToCopy.add(update);


        // Create loop control initializer
        ComplexExpressionNode init = (ComplexExpressionNode) node.getChildren().get(0).deepCopy();
        init.getExpression().replaceDataElement(node.getLoopControlVariable(), localTable.get(iterationData.getIdentifier()));
        init.setParent(node.getParent());
        init.setVariableTable(localTable);

        result.add(init);

        //copy children n times
        for (int i = 0; i < numIterations; i++) {
            for (PatternNode current : childrenToCopy) {
                PatternNode currentNode = current.deepCopy();
                currentNode.setParent(node.getParent());
                if (currentNode instanceof ForLoopNode) {
                    localTable.put(((ForLoopNode) currentNode).getLoopControlVariable().getIdentifier(), ((ForLoopNode) currentNode).getLoopControlVariable());
                } else if (currentNode instanceof ForEachLoopNode) {
                    localTable.put(((ForEachLoopNode) currentNode).getLoopControlVariable().getIdentifier(), ((ForEachLoopNode) currentNode).getLoopControlVariable());
                }
                HashMap<String, Data> oldTable = currentNode.getVariableTable();
                currentNode.setVariableTable(localTable);
                localTable = new HashMap<>(localTable);
                oldTable.entrySet().stream().filter(x -> !currentNode.getParent().getVariableTable().containsKey(x)).filter(x -> !x.getValue().isInitialized()).forEach(x -> currentNode.getVariableTable().put(x.getKey(), x.getValue()));
                result.add(currentNode);
            }
        }
        localTable.remove(node.getLoopControlVariable().getIdentifier());
        node.getParent().setVariableTable(localTable);

        DeepCopyHelper.removeScope();

        return result;
    }


    /**
     * Subclass to simplify the replacement of variables.
     */
    private static class FindAndReplaceLoopVariable implements APTVisitor {

        private final Data originalVar;

        private final Data newVar;

        private final PatternNode startNode;

        public FindAndReplaceLoopVariable(Data originalVar, Data newVar, PatternNode startNode) {
            this.originalVar = originalVar;
            this.newVar = newVar;
            this.startNode = startNode;
        }

        public void doReplace() {
            startNode.accept(this.getRealThis());
        }

        @Override
        public void traverse(ParallelCallNode node) {
            node.getChildren().get(0).accept(this.getRealThis());
        }

        @Override
        public void traverse(CallNode node) {

        }

        @Override
        public void visit(ComplexExpressionNode node) {
            node.getExpression().replaceDataElement(originalVar, newVar);
            node.getVariableTable().put(newVar.getIdentifier(), newVar);
        }

        @Override
        public void visit(SimpleExpressionBlockNode node) {
            for (IRLExpression exp : node.getExpressionList()) {
                exp.replaceDataElement(originalVar, newVar);
            }
            node.getVariableTable().put(newVar.getIdentifier(), newVar);
        }
    }
}
