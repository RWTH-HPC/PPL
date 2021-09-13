package LoopIterationEstimator;

import de.parallelpatterndsl.patterndsl.abstractPatternTree.Nodes.PatternNode;
import de.parallelpatterndsl.patterndsl.abstractPatternTree.Nodes.Plain.ForLoopNode;
import de.parallelpatterndsl.patterndsl.abstractPatternTree.Nodes.Plain.SimpleExpressionBlockNode;
import de.parallelpatterndsl.patterndsl.abstractPatternTree.Visitor.APTVisitor;
import de.parallelpatterndsl.patterndsl.expressions.IRLExpression;

public class FLOPTest implements APTVisitor {

    private int flops;

    private int multiplier;

    public FLOPTest() {
        this.flops = 0;
        this.multiplier = 1;
    }

    public int estimate(PatternNode node) {
        if (flops == 0) {
            node.accept(this.getRealThis());
        }
        return flops;
    }

    @Override
    public void visit(SimpleExpressionBlockNode node) {
        for (IRLExpression expression : node.getExpressionList()) {
            this.flops += multiplier * expression.getOperationCount();
        }
    }

    @Override
    public void visit(ForLoopNode node) {
        multiplier *= node.getNumIterations();
    }

    @Override
    public void endVisit(ForLoopNode node) {
        multiplier = multiplier / node.getNumIterations();
    }

}
