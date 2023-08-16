package de.parallelpatterndsl.patterndsl.abstractPatternTree.Nodes.Plain;

import de.parallelpatterndsl.patterndsl.abstractPatternTree.Nodes.PatternNode;
import de.parallelpatterndsl.patterndsl.abstractPatternTree.Visitor.APTVisitor;
import de.parallelpatterndsl.patterndsl.expressions.AssignmentExpression;
import de.parallelpatterndsl.patterndsl.expressions.IRLExpression;

/**
 * Abstract definition of a loop for the abstract pattern tree.
 */
public abstract class LoopNode extends PatternNode {

    public LoopNode() {
    }

    /**
     * returns the number of iteration done within the loop.
     * @return
     */
    public abstract int getNumIterations();

    @Override
    public long getCost() {
        long cost = 0;
        for (PatternNode child: getChildren() ) {
            cost += child.getCost();
        }
        return cost * getNumIterations();
    }

    @Override
    public long getLoadStore() {
        long cost = 0;
        for (PatternNode child: getChildren() ) {
            cost += child.getLoadStore();
        }
        return cost * getNumIterations();
    }

    public boolean isSimple() {
        boolean result = false;

        if (this instanceof ForLoopNode) {
            FindLoopSkip finder = new FindLoopSkip((ForLoopNode) this);
            WriteAccessTester tester = new WriteAccessTester((ForLoopNode) this);
            if (!finder.find() && !tester.test() && ((ForLoopNode) this).hasSimpleLength()) {
                result = true;
            }
        }

        return result;
    }

    private static class FindLoopSkip implements APTVisitor {
        private boolean found = false;

        private final ForLoopNode startNode;

        public FindLoopSkip(ForLoopNode startNode) {
            this.startNode = startNode;
        }

        public boolean find(){
            startNode.accept(this.getRealThis());
            return found;
        }

        @Override
        public void endVisit(LoopSkipNode node) {
            found = true;
        }
    }

    private static class WriteAccessTester implements APTVisitor{
        private boolean writeAccess = false;

        private final ForLoopNode startNode;

        public WriteAccessTester(ForLoopNode startNode) {
            this.startNode = startNode;
        }

        public boolean test(){
            for (int i = 3; i < startNode.getChildren().size(); i++) {
                PatternNode currentNode = startNode.getChildren().get(i);
                currentNode.accept(this.getRealThis());
            }
            return writeAccess;
        }

        @Override
        public void visit(ComplexExpressionNode node) {
            if (node.getExpression() instanceof AssignmentExpression) {
                if (((AssignmentExpression) node.getExpression()).getOutputElement() == startNode.getLoopControlVariable()) {
                    writeAccess = true;
                }
            }
        }

        @Override
        public void visit(SimpleExpressionBlockNode node) {
            for (IRLExpression exp: node.getExpressionList() ) {
                if (exp instanceof AssignmentExpression) {
                    if (((AssignmentExpression) exp).getOutputElement() == startNode.getLoopControlVariable()) {
                        writeAccess = true;
                    }
                }
            }
        }
    }
}
