package de.parallelpatterndsl.patterndsl.abstractPatternTree.Visitor;

import de.parallelpatterndsl.patterndsl.abstractPatternTree.AbstractPatternTree;
import de.parallelpatterndsl.patterndsl.abstractPatternTree.Nodes.FunctionNode;
import de.parallelpatterndsl.patterndsl.abstractPatternTree.Nodes.Plain.CallNode;
import de.parallelpatterndsl.patterndsl.abstractPatternTree.Nodes.Plain.ParallelCallNode;
import de.parallelpatterndsl.patterndsl.helperLibrary.PredefinedFunctions;

/**
 * Implements a Visitor to reset the call counter after using the extended Shape Visitor.
 */
public class CallCountResetter implements APTVisitor{

    @Override
    public void visit(ParallelCallNode node) {
        node.resetCallCount();
        FunctionNode function = AbstractPatternTree.getFunctionTable().get(node.getFunctionIdentifier());
        function.resetCurrentCall();
    }

    @Override
    public void visit(CallNode node) {
        node.resetCallCount();
        if (!PredefinedFunctions.contains(node.getFunctionIdentifier())) {
            FunctionNode function = AbstractPatternTree.getFunctionTable().get(node.getFunctionIdentifier());
            function.resetCurrentCall();
        }
    }

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
