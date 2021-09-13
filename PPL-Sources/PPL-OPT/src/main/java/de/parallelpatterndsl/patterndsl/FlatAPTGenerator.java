package de.parallelpatterndsl.patterndsl;

import de.parallelpatterndsl.patterndsl.abstractPatternTree.AbstractPatternTree;
import de.parallelpatterndsl.patterndsl.abstractPatternTree.Nodes.Functions.SerialNode;
import de.parallelpatterndsl.patterndsl.abstractPatternTree.Nodes.PatternNode;
import de.parallelpatterndsl.patterndsl.abstractPatternTree.Nodes.Plain.*;
import de.parallelpatterndsl.patterndsl.abstractPatternTree.Visitor.APTVisitor;

import java.util.ArrayList;

/**
 * Generator constructing a FlatAPT from an APT.
 */
public class FlatAPTGenerator implements APTVisitor {

    private static FlatAPTGenerator factory = new FlatAPTGenerator();

    private FlatAPT flatAPT;

    private FlatAPTGenerator() {}

    /**
     * Generates a FlatAPT from an APT.
     * @param apt - AbstractPatternTree to be analyzed.
     * @param splitSize - hyperparameter job length.
     * @param dataSplitSize - hyperparameter network
     * @return flatAPT
     */
    public static FlatAPT generate(AbstractPatternTree apt, int splitSize, int dataSplitSize) {
        OPTReturnDependencyHandler returnDependencyHandler = new OPTReturnDependencyHandler();
        apt.getRoot().accept(returnDependencyHandler);

        factory.flatAPT = new FlatAPT(splitSize, dataSplitSize);
        apt.getRoot().accept(factory.getRealThis());

        FlatAPT table = factory.flatAPT;
        factory.flatAPT = null;

        return table;
    }

    @Override
    public void traverse(ComplexExpressionNode node) {}

    @Override
    public void traverse(ForLoopNode node) { }

    @Override
    public void traverse(ForEachLoopNode node) { }

    @Override
    public void traverse(WhileLoopNode node) { }

    @Override
    public void traverse(BranchNode node) { }

    @Override
    public void endVisit(SerialNode node) {
        // Call only for main.
        for (PatternNode child : node.getChildren()) {
            this.flatAPT.add(child);
        }
    }


    /**
     * Simple class to add all dependencies to a return node in order to cover the step table creation.
     */
    private static class OPTReturnDependencyHandler implements APTVisitor {


        @Override
        public void visit(ReturnNode node) {
            node.setInputElements(new ArrayList<>(node.getVariableTable().values()));
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

}
