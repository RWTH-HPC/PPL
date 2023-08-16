package de.parallelpatterndsl.patterndsl.MappingTree.Visitor;

import de.parallelpatterndsl.patterndsl.MappingTree.Nodes.DataControl.*;
import de.parallelpatterndsl.patterndsl.MappingTree.Nodes.Function.*;
import de.parallelpatterndsl.patterndsl.MappingTree.Nodes.MappingNode;
import de.parallelpatterndsl.patterndsl.MappingTree.Nodes.ParallelCalls.*;
import de.parallelpatterndsl.patterndsl.MappingTree.Nodes.Plain.*;

public interface AMTVisitor {

    /**
     * RealThis is used to allow visitor composition, where a delegating visitor
     * utilizes this setter to set another visitor as the handle/traversal
     * controller. If this method is not overridden by the language developer,
     * the visitor still can be reused, by implementing this method in a
     * decorator.
     * @param realThis the real instance to use for handling and traversing nodes.
     */
    default public void setRealThis(AMTVisitor realThis) {
        throw new UnsupportedOperationException("0xA7011x408 The setter for realThis is not implemented. You might want to implement a wrapper class to allow setting/getting realThis.");
    }

    /**
     * By default this method returns {@code this}. Visitors intended for reusage
     * in other languages should override this method together with
     * to make a visitor
     * composable.
     */
    default public AMTVisitor getRealThis() {
        return (AMTVisitor) this;
    }



    public default void visit(BranchCaseMapping node) {}

    public default void endVisit(BranchCaseMapping node) {}

    public default void traverse(BranchCaseMapping node) {
        if (node.hasCondition()) {
            node.getCondition().accept(getRealThis());
        }
        for (MappingNode child: node.getChildren() ) {
            child.accept(getRealThis());
        }
    }

    public default void handle(BranchCaseMapping node) {
        visit(node);
        traverse(node);
        endVisit(node);
    }

    public default void visit(BranchMapping node) {}

    public default void endVisit(BranchMapping node) {}

    public default void traverse(BranchMapping node) {
        for (MappingNode child: node.getChildren() ) {
            child.accept(getRealThis());
        }
    }

    public default void handle(BranchMapping node) {
        visit(node);
        traverse(node);
        endVisit(node);
    }

    public default void visit(CallMapping node) {}

    public default void endVisit(CallMapping node) {}

    public default void traverse(CallMapping node) {
        for (MappingNode child: node.getChildren() ) {
            child.accept(getRealThis());
        }
    }

    public default void handle(CallMapping node) {
        visit(node);
        traverse(node);
        endVisit(node);
    }

    public default void handle(ComplexExpressionMapping node) {
        visit(node);
        traverse(node);
        endVisit(node);
    }

    public default void visit(ComplexExpressionMapping node) {}

    public default void endVisit(ComplexExpressionMapping node) {}

    public default void traverse(ComplexExpressionMapping node) {
        for (MappingNode child: node.getChildren() ) {
            child.accept(getRealThis());
        }
    }

    public default void handle(LoopSkipMapping node) {
        visit(node);
        traverse(node);
        endVisit(node);
    }

    public default void visit(LoopSkipMapping node) {}

    public default void endVisit(LoopSkipMapping node) {}

    public default void traverse(LoopSkipMapping node) {
        for (MappingNode child: node.getChildren() ) {
            child.accept(getRealThis());
        }
    }

    public default void visit(ForEachLoopMapping node) {}

    public default void endVisit(ForEachLoopMapping node) {}

    public default void traverse(ForEachLoopMapping node) {
        node.getParsedList().accept(getRealThis());
        for (MappingNode child: node.getChildren() ) {
            child.accept(getRealThis());
        }
    }

    public default void handle(ForEachLoopMapping node) {
        visit(node);
        traverse(node);
        endVisit(node);
    }

    public default void visit(ForLoopMapping node) {}

    public default void endVisit(ForLoopMapping node) {}

    public default void traverse(ForLoopMapping node) {
        node.getInitExpression().accept(getRealThis());
        node.getControlExpression().accept(getRealThis());
        node.getUpdateExpression().accept(getRealThis());
        for (MappingNode child: node.getChildren() ) {
            child.accept(getRealThis());
        }
    }

    public default void handle(ForLoopMapping node) {
        visit(node);
        traverse(node);
        endVisit(node);
    }


    public default void visit(ReturnMapping node) {}

    public default void endVisit(ReturnMapping node) {}

    public default void traverse(ReturnMapping node) {
        if (node.getResult().isPresent()) {
            node.getResult().get().accept(getRealThis());
        }
        for (MappingNode child: node.getChildren() ) {
            child.accept(getRealThis());
        }
    }

    public default void handle(ReturnMapping node) {
        visit(node);
        traverse(node);
        endVisit(node);
    }


    public default void visit(SimpleExpressionBlockMapping node) {}

    public default void endVisit(SimpleExpressionBlockMapping node) {}

    public default void traverse(SimpleExpressionBlockMapping node) {
        for (MappingNode child: node.getChildren() ) {
            child.accept(getRealThis());
        }
    }

    public default void handle(SimpleExpressionBlockMapping node) {
        visit(node);
        traverse(node);
        endVisit(node);
    }


    public default void visit(WhileLoopMapping node) {}

    public default void endVisit(WhileLoopMapping node) {}

    public default void traverse(WhileLoopMapping node) {
        node.getCondition().accept(getRealThis());
        for (MappingNode child: node.getChildren() ) {
            child.accept(getRealThis());
        }
    }

    public default void handle(WhileLoopMapping node) {
        visit(node);
        traverse(node);
        endVisit(node);
    }

    public default void visit(JumpLabelMapping node) {}

    public default void endVisit(JumpLabelMapping node) {}

    public default void traverse(JumpLabelMapping node) {
        for (MappingNode child: node.getChildren() ) {
            child.accept(getRealThis());
        }
    }

    public default void handle(JumpLabelMapping node) {
        visit(node);
        traverse(node);
        endVisit(node);
    }


    public default void visit(JumpStatementMapping node) {}

    public default void endVisit(JumpStatementMapping node) {}

    public default void traverse(JumpStatementMapping node) {
        node.getResultExpression().accept(getRealThis());
        for (MappingNode child: node.getChildren() ) {
            child.accept(getRealThis());
        }
    }

    public default void handle(JumpStatementMapping node) {
        visit(node);
        traverse(node);
        endVisit(node);
    }


    /************************************************************
     *
     *
     * Parallel Calls
     *
     *
     ************************************************************/

    public default void visit(FusedParallelCallMapping node) {}

    public default void endVisit(FusedParallelCallMapping node) {}

    public default void traverse(FusedParallelCallMapping node) {
        for (MappingNode child: node.getChildren() ) {
            child.accept(getRealThis());
        }
    }

    public default void handle(FusedParallelCallMapping node) {
        visit(node);
        traverse(node);
        endVisit(node);
    }


    public default void visit(GPUParallelCallMapping node) {}

    public default void endVisit(GPUParallelCallMapping node) {}

    public default void traverse(GPUParallelCallMapping node) {
        node.getDefinition().accept(getRealThis());
        for (MappingNode child: node.getChildren() ) {
            child.accept(getRealThis());
        }

        if (node.getDynamicProgrammingBarrier().isPresent()) {
            node.getDynamicProgrammingBarrier().get().accept(getRealThis());
        }

        for (AbstractDataMovementMapping child: node.getDpPreSwapTransfers() ) {
            child.accept(getRealThis());
        }

        for (AbstractDataMovementMapping child: node.getDynamicProgrammingdataTransfers() ) {
            child.accept(getRealThis());
        }

        for (AbstractDataMovementMapping child: node.getDpPostSwapTransfers() ) {
            child.accept(getRealThis());
        }
    }

    public default void handle(GPUParallelCallMapping node) {
        visit(node);
        traverse(node);
        endVisit(node);
    }


    public default void visit(ParallelCallMapping node) {}

    public default void endVisit(ParallelCallMapping node) {}

    public default void traverse(ParallelCallMapping node) {
        node.getDefinition().accept(getRealThis());
        for (MappingNode child: node.getChildren() ) {
            child.accept(getRealThis());
        }

        if (node.getDynamicProgrammingBarrier().isPresent()) {
            node.getDynamicProgrammingBarrier().get().accept(getRealThis());
        }

        for (AbstractDataMovementMapping child: node.getDpPreSwapTransfers() ) {
            child.accept(getRealThis());
        }

        for (AbstractDataMovementMapping child: node.getDynamicProgrammingdataTransfers() ) {
            child.accept(getRealThis());
        }

        for (AbstractDataMovementMapping child: node.getDpPostSwapTransfers() ) {
            child.accept(getRealThis());
        }
    }

    public default void handle(ParallelCallMapping node) {
        visit(node);
        traverse(node);
        endVisit(node);
    }


    public default void visit(ReductionCallMapping node) {}

    public default void endVisit(ReductionCallMapping node) {}

    public default void traverse(ReductionCallMapping node) {
        node.getDefinition().accept(getRealThis());
        for (MappingNode child: node.getChildren() ) {
            child.accept(getRealThis());
        }
    }

    public default void handle(ReductionCallMapping node) {
        visit(node);
        traverse(node);
        endVisit(node);
    }

    public default void visit(SerializedParallelCallMapping node) {}

    public default void endVisit(SerializedParallelCallMapping node) {}

    public default void traverse(SerializedParallelCallMapping node) {
        node.getDefinition().accept(getRealThis());
        for (MappingNode child: node.getChildren() ) {
            child.accept(getRealThis());
        }
    }

    public default void handle(SerializedParallelCallMapping node) {
        visit(node);
        traverse(node);
        endVisit(node);
    }


    /************************************************************
     *
     *
     * Functions
     *
     *
     ************************************************************/


    public default void visit(DynamicProgrammingMapping node) {}

    public default void endVisit(DynamicProgrammingMapping node) {}

    public default void traverse(DynamicProgrammingMapping node) {
        for (MappingNode child: node.getChildren() ) {
            child.accept(getRealThis());
        }
    }

    public default void handle(DynamicProgrammingMapping node) {
        visit(node);
        traverse(node);
        endVisit(node);
    }


    public default void visit(MainMapping node) {}

    public default void endVisit(MainMapping node) {}

    public default void traverse(MainMapping node) {
        for (MappingNode child: node.getChildren() ) {
            child.accept(getRealThis());
        }
    }

    public default void handle(MainMapping node) {
        visit(node);
        traverse(node);
        endVisit(node);
    }

    public default void visit(MapMapping node) {}

    public default void endVisit(MapMapping node) {}

    public default void traverse(MapMapping node) {
        for (MappingNode child: node.getChildren() ) {
            child.accept(getRealThis());
        }
    }

    public default void handle(MapMapping node) {
        visit(node);
        traverse(node);
        endVisit(node);
    }


    public default void visit(RecursionMapping node) {}

    public default void endVisit(RecursionMapping node) {}

    public default void traverse(RecursionMapping node) {
        for (MappingNode child: node.getChildren() ) {
            child.accept(getRealThis());
        }
    }

    public default void handle(RecursionMapping node) {
        visit(node);
        traverse(node);
        endVisit(node);
    }


    public default void visit(ReduceMapping node) {}

    public default void endVisit(ReduceMapping node) {}

    public default void traverse(ReduceMapping node) {
        for (MappingNode child: node.getChildren() ) {
            child.accept(getRealThis());
        }
    }

    public default void handle(ReduceMapping node) {
        visit(node);
        traverse(node);
        endVisit(node);
    }


    public default void visit(SerialMapping node) {}

    public default void endVisit(SerialMapping node) {}

    public default void traverse(SerialMapping node) {
        for (MappingNode child: node.getChildren() ) {
            child.accept(getRealThis());
        }
    }

    public default void handle(SerialMapping node) {
        visit(node);
        traverse(node);
        endVisit(node);
    }


    public default void visit(StencilMapping node) {}

    public default void endVisit(StencilMapping node) {}

    public default void traverse(StencilMapping node) {
        for (MappingNode child: node.getChildren() ) {
            child.accept(getRealThis());
        }
    }

    public default void handle(StencilMapping node) {
        visit(node);
        traverse(node);
        endVisit(node);
    }


    /************************************************************
     *
     *
     * Data Control Nodes
     *
     *
     ************************************************************/


    public default void visit(BarrierMapping node) {}

    public default void endVisit(BarrierMapping node) {}

    public default void traverse(BarrierMapping node) {
        for (MappingNode child: node.getChildren() ) {
            child.accept(getRealThis());
        }
    }

    public default void handle(BarrierMapping node) {
        visit(node);
        traverse(node);
        endVisit(node);
    }


    public default void visit(DataMovementMapping node) {}

    public default void endVisit(DataMovementMapping node) {}

    public default void traverse(DataMovementMapping node) {
        for (MappingNode child: node.getChildren() ) {
            child.accept(getRealThis());
        }
    }

    public default void handle(DataMovementMapping node) {
        visit(node);
        traverse(node);
        endVisit(node);
    }


    public default void visit(GPUDeAllocationMapping node) {}

    public default void endVisit(GPUDeAllocationMapping node) {}

    public default void traverse(GPUDeAllocationMapping node) {
        for (MappingNode child: node.getChildren() ) {
            child.accept(getRealThis());
        }
    }

    public default void handle(GPUDeAllocationMapping node) {
        visit(node);
        traverse(node);
        endVisit(node);
    }

    public default void visit(GPUDataMovementMapping node) {}

    public default void endVisit(GPUDataMovementMapping node) {}

    public default void traverse(GPUDataMovementMapping node) {
        for (MappingNode child: node.getChildren() ) {
            child.accept(getRealThis());
        }
    }

    public default void handle(GPUDataMovementMapping node) {
        visit(node);
        traverse(node);
        endVisit(node);
    }

    public default void visit(GPUAllocationMapping node) {}

    public default void endVisit(GPUAllocationMapping node) {}

    public default void traverse(GPUAllocationMapping node) {
        for (MappingNode child: node.getChildren() ) {
            child.accept(getRealThis());
        }
    }

    public default void handle(GPUAllocationMapping node) {
        visit(node);
        traverse(node);
        endVisit(node);
    }

}
