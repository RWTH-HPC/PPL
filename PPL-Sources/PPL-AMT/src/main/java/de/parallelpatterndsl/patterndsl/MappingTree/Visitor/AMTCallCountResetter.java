package de.parallelpatterndsl.patterndsl.MappingTree.Visitor;

import de.parallelpatterndsl.patterndsl.MappingTree.Nodes.ParallelCalls.GPUParallelCallMapping;
import de.parallelpatterndsl.patterndsl.MappingTree.Nodes.ParallelCalls.ParallelCallMapping;
import de.parallelpatterndsl.patterndsl.MappingTree.Nodes.ParallelCalls.ReductionCallMapping;
import de.parallelpatterndsl.patterndsl.MappingTree.Nodes.Plain.CallMapping;
import de.parallelpatterndsl.patterndsl.abstractPatternTree.AbstractPatternTree;
import de.parallelpatterndsl.patterndsl.abstractPatternTree.Nodes.FunctionNode;

/**
 * Implements a Visitor to reset the call counter after using the extended Shape Visitor.
 */
public class AMTCallCountResetter implements AMTVisitor{

    @Override
    public void visit(ParallelCallMapping node) {
        node.resetCallCount();
        FunctionNode function = AbstractPatternTree.getFunctionTable().get(node.getFunctionIdentifier());
        function.resetCurrentCall();
    }

    @Override
    public void visit(GPUParallelCallMapping node) {
        node.resetCallCount();
        FunctionNode function = AbstractPatternTree.getFunctionTable().get(node.getFunctionIdentifier());
        function.resetCurrentCall();
    }

    @Override
    public void visit(ReductionCallMapping node) {
        node.resetCallCount();
        FunctionNode function = AbstractPatternTree.getFunctionTable().get(node.getFunctionIdentifier());
        function.resetCurrentCall();
    }

    @Override
    public void visit(CallMapping node) {
        node.resetCallCount();
        if (!node.getFunctionIdentifier().equals("init_List")) {
            FunctionNode function = AbstractPatternTree.getFunctionTable().get(node.getFunctionIdentifier());
            function.resetCurrentCall();
        }
    }

    private AMTVisitor realThis = this;

    @Override
    public AMTVisitor getRealThis() {
        return realThis;
    }

    @Override
    public void setRealThis(AMTVisitor realThis) {
        this.realThis = realThis;
    }
}
