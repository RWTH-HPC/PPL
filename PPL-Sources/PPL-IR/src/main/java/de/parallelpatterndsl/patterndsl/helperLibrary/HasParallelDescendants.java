package de.parallelpatterndsl.patterndsl.helperLibrary;

import de.parallelpatterndsl.patterndsl.abstractPatternTree.Nodes.PatternNode;
import de.parallelpatterndsl.patterndsl.abstractPatternTree.Nodes.Plain.ParallelCallNode;
import de.parallelpatterndsl.patterndsl.abstractPatternTree.Visitor.APTVisitor;

public class HasParallelDescendants implements APTVisitor {
    private boolean result = false;

    public boolean getResult(PatternNode node) {
        result = false;

        node.accept(getRealThis());

        return result;
    }

    @Override
    public void visit(ParallelCallNode node) {
        result = true;
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