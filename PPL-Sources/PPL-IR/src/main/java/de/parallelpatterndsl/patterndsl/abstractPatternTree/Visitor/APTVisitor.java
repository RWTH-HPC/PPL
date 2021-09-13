package de.parallelpatterndsl.patterndsl.abstractPatternTree.Visitor;


import de.parallelpatterndsl.patterndsl.abstractPatternTree.Nodes.Functions.*;
import de.parallelpatterndsl.patterndsl.abstractPatternTree.Nodes.PatternNode;
import de.parallelpatterndsl.patterndsl.abstractPatternTree.Nodes.Plain.*;

/**
 * Interface to allow a simple traversal over the whole abstract pattern tree.
 */
public interface APTVisitor {

    /**
     * RealThis is used to allow visitor composition, where a delegating visitor
     * utilizes this setter to set another visitor as the handle/traversal
     * controller. If this method is not overridden by the language developer,
     * the visitor still can be reused, by implementing this method in a
     * decorator.
     * @param realThis the real instance to use for handling and traversing nodes.
     */
    default public void setRealThis(APTVisitor realThis) {
        throw new UnsupportedOperationException("0xA7011x408 The setter for realThis is not implemented. You might want to implement a wrapper class to allow setting/getting realThis.");
    }

    /**
     * By default this method returns {@code this}. Visitors intended for reusage
     * in other languages should override this method together with
     * to make a visitor
     * composable.
     */
    default public APTVisitor getRealThis() {
        return (APTVisitor) this;
    }

    /**
     *
     *
     * Plain nodes.
     *
     */


    default public void visit(SimpleExpressionBlockNode node) {}

    default public void endVisit(SimpleExpressionBlockNode node){}

    default public void handle(SimpleExpressionBlockNode node){
        getRealThis().visit(node);
            getRealThis().traverse(node);
        getRealThis().endVisit(node);
    }

    default public void traverse(SimpleExpressionBlockNode node){}


    default public void visit(ParallelCallNode node) {}

    default public void endVisit(ParallelCallNode node){}

    default public void handle(ParallelCallNode node){
        getRealThis().visit(node);
        getRealThis().traverse(node);
        getRealThis().endVisit(node);
    }

    default public void traverse(ParallelCallNode node){
        for (PatternNode child: node.getChildren()) {
            child.accept(getRealThis());
        }
    }


    default public void visit(ForLoopNode node) {}

    default public void endVisit(ForLoopNode node){}

    default public void handle(ForLoopNode node){
        getRealThis().visit(node);
        getRealThis().traverse(node);
        getRealThis().endVisit(node);
    }

    default public void traverse(ForLoopNode node){
        for (PatternNode child: node.getChildren()) {
            child.accept(getRealThis());
        }
    }




    default public void visit(ComplexExpressionNode node) {}

    default public void endVisit(ComplexExpressionNode node){}

    default public void handle(ComplexExpressionNode node){
        getRealThis().visit(node);
        getRealThis().traverse(node);
        getRealThis().endVisit(node);
    }

    default public void traverse(ComplexExpressionNode node){
        for (PatternNode child: node.getChildren()) {
            child.accept(getRealThis());
        }
    }


    default public void visit(CallNode node) {}

    default public void endVisit(CallNode node){}

    default public void handle(CallNode node){
        getRealThis().visit(node);
        getRealThis().traverse(node);
        getRealThis().endVisit(node);
    }

    default public void traverse(CallNode node){
        for (PatternNode child: node.getChildren()) {
            child.accept(getRealThis());
        }
    }


    default public void visit(BranchNode node) {}

    default public void endVisit(BranchNode node){}

    default public void handle(BranchNode node){
        getRealThis().visit(node);
        getRealThis().traverse(node);
        getRealThis().endVisit(node);
    }

    default public void traverse(BranchNode node){
        for (PatternNode child: node.getChildren()) {
            child.accept(getRealThis());
        }
    }


    default public void visit(BranchCaseNode node) {}

    default public void endVisit(BranchCaseNode node){}

    default public void handle(BranchCaseNode node){
        getRealThis().visit(node);
        getRealThis().traverse(node);
        getRealThis().endVisit(node);
    }

    default public void traverse(BranchCaseNode node){
        for (PatternNode child: node.getChildren()) {
            child.accept(getRealThis());
        }
    }


    default public void visit(StencilNode node) {}

    default public void endVisit(StencilNode node){}

    default public void handle(StencilNode node){
        getRealThis().visit(node);
        getRealThis().traverse(node);
        getRealThis().endVisit(node);
    }

    default public void traverse(StencilNode node){
        for (PatternNode child: node.getChildren()) {
            child.accept(getRealThis());
        }
    }


    default public void visit(SerialNode node) {}

    default public void endVisit(SerialNode node){}

    default public void handle(SerialNode node){
        getRealThis().visit(node);
        getRealThis().traverse(node);
        getRealThis().endVisit(node);
    }

    default public void traverse(SerialNode node){
        for (PatternNode child: node.getChildren()) {
            child.accept(getRealThis());
        }
    }


    default public void visit(ReduceNode node) {}

    default public void endVisit(ReduceNode node){}

    default public void handle(ReduceNode node){
        getRealThis().visit(node);
        getRealThis().traverse(node);
        getRealThis().endVisit(node);
    }

    default public void traverse(ReduceNode node){
        for (PatternNode child: node.getChildren()) {
            child.accept(getRealThis());
        }
    }


    default public void visit(MapNode node) {}

    default public void endVisit(MapNode node){}

    default public void handle(MapNode node){
        getRealThis().visit(node);
        getRealThis().traverse(node);
        getRealThis().endVisit(node);
    }

    default public void traverse(MapNode node){
        for (PatternNode child: node.getChildren()) {
            child.accept(getRealThis());
        }
    }



    default public void visit(WhileLoopNode node) {}

    default public void endVisit(WhileLoopNode node){}

    default public void handle(WhileLoopNode node) {
        getRealThis().visit(node);
        getRealThis().traverse(node);
        getRealThis().endVisit(node);
    }

    default public void traverse(WhileLoopNode node){
        for (PatternNode child: node.getChildren()) {
            child.accept(getRealThis());
        }
    }



    default public void visit(ForEachLoopNode node) {}

    default public void endVisit(ForEachLoopNode node){}

    default public void handle(ForEachLoopNode node) {
        getRealThis().visit(node);
        getRealThis().traverse(node);
        getRealThis().endVisit(node);
    }

    default public void traverse(ForEachLoopNode node){
        for (PatternNode child: node.getChildren()) {
            child.accept(getRealThis());
        }
    }

    default public void visit(ReturnNode node) {}

    default public void endVisit(ReturnNode node){}

    default public void handle(ReturnNode node) {
        getRealThis().visit(node);
        getRealThis().traverse(node);
        getRealThis().endVisit(node);
    }

    default public void traverse(ReturnNode node){
        for (PatternNode child: node.getChildren()) {
            child.accept(getRealThis());
        }
    }

    default public void visit(RecursionNode node) {}

    default public void endVisit(RecursionNode node){}

    default public void handle(RecursionNode node) {
        getRealThis().visit(node);
        getRealThis().traverse(node);
        getRealThis().endVisit(node);
    }

    default public void traverse(RecursionNode node){
        for (PatternNode child: node.getChildren()) {
            child.accept(getRealThis());
        }
    }

    default public void visit(DynamicProgrammingNode node) {}

    default public void endVisit(DynamicProgrammingNode node){}

    default public void handle(DynamicProgrammingNode node) {
        getRealThis().visit(node);
        getRealThis().traverse(node);
        getRealThis().endVisit(node);
    }

    default public void traverse(DynamicProgrammingNode node){
        for (PatternNode child: node.getChildren()) {
            child.accept(getRealThis());
        }
    }


    default public void visit(JumpLabelNode node) {}

    default public void endVisit(JumpLabelNode node){}

    default public void handle(JumpLabelNode node) {
        getRealThis().visit(node);
        getRealThis().traverse(node);
        getRealThis().endVisit(node);
    }

    default public void traverse(JumpLabelNode node){
        for (PatternNode child: node.getChildren()) {
            child.accept(getRealThis());
        }
    }


    default public void visit(JumpStatementNode node) {}

    default public void endVisit(JumpStatementNode node){}

    default public void handle(JumpStatementNode node) {
        getRealThis().visit(node);
        getRealThis().traverse(node);
        getRealThis().endVisit(node);
    }

    default public void traverse(JumpStatementNode node){
        for (PatternNode child: node.getChildren()) {
            child.accept(getRealThis());
        }
    }

}
