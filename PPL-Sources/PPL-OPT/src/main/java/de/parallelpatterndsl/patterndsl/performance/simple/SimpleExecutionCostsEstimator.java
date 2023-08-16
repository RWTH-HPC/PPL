package de.parallelpatterndsl.patterndsl.performance.simple;

import de.parallelpatterndsl.patterndsl.abstractPatternTree.AbstractPatternTree;
import de.parallelpatterndsl.patterndsl.abstractPatternTree.Nodes.Functions.ReduceNode;
import de.parallelpatterndsl.patterndsl.abstractPatternTree.Nodes.PatternNode;
import de.parallelpatterndsl.patterndsl.abstractPatternTree.Nodes.Plain.*;
import de.parallelpatterndsl.patterndsl.abstractPatternTree.Nodes.Plain.AdditionalArguments.MetaList;
import de.parallelpatterndsl.patterndsl.abstractPatternTree.Nodes.Plain.AdditionalArguments.MetaValue;
import de.parallelpatterndsl.patterndsl.abstractPatternTree.Visitor.APTVisitor;
import de.parallelpatterndsl.patterndsl.expressions.IRLExpression;
import de.parallelpatterndsl.patterndsl.hardwareDescription.Processor;
import de.parallelpatterndsl.patterndsl.patternSplits.PatternSplit;
import de.parallelpatterndsl.patterndsl.patternSplits.SerialPatternSplit;
import de.parallelpatterndsl.patterndsl.performance.ExecutionCostsEstimator;
import de.parallelpatterndsl.patterndsl.teams.Team;

import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Set;
import java.util.stream.Collectors;

/**
 * Estimates the FLOPS of a PatternNode and its children using the APTVisitor.
 */
public class SimpleExecutionCostsEstimator implements ExecutionCostsEstimator {

    @Override
    public double estimate(PatternSplit split, Team team) {
        double costs = this.estimateNode(split.getNode(), team.getProcessor());
        long length = 1;
        for (long l : split.getLengths()) {
            length *= l;
        }

        costs = costs * length / Long.min(length, team.getCores());
        costs = costs / (team.getProcessor().getArithmeticUnits() * team.getProcessor().getFrequency() * FREQUENCY_TO_SECONDS);
        return costs;
    }

    @Override
    public double estimateAll(Set<PatternSplit> splits, Team team) {
        Map<PatternNode, List<PatternSplit>> groupBy = splits.stream().collect(Collectors.groupingBy(PatternSplit::getNode));
        double costs = 0.0;
        for (PatternNode node : groupBy.keySet()) {
            double c = this.estimateNode(node, team.getProcessor());
            long length = 1;
            for (PatternSplit split : groupBy.get(node)) {
                double ll = 1;
                for (long l : split.getLengths()) {
                    ll *= l;
                }
                length += ll;
            }

            c = c * length / Long.min(length, team.getCores());
            c = c / (team.getProcessor().getArithmeticUnits() * team.getProcessor().getFrequency() * FREQUENCY_TO_SECONDS);
            costs += c;
        }

        return costs;
    }

    private long flops;

    private long multiplier;

    private Processor processor;

    private PatternNode node;

    private static final int WARP_PENALTY = 32;

    public long estimateNode(PatternNode node, Processor processor) {
        this.flops = 0;
        this.multiplier = 1;
        this.node = node;
        this.processor = processor;

        node.accept(this.getRealThis());
        return this.flops;
    }

    @Override
    public void traverse(ParallelCallNode node) {
        if (node == this.node) {
            for (PatternNode child : node.getChildren()) {
                child.accept(this.getRealThis());
            }
        } else {
            SimpleExecutionCostsEstimator estimator = new SimpleExecutionCostsEstimator();
            long costs = estimator.estimateNode(node, this.processor);

            long[] lengths = new SimpleParallelismEstimator().estimate(node);
            for (long length : lengths) {
                this.flops += costs * length;
            }
        }

        if (AbstractPatternTree.getFunctionTable().get(node.getFunctionIdentifier()) instanceof ReduceNode) {
            //this.flops -= 4096;//((MetaList<Long>) node.getAdditionalArguments().get(0)).getValues().get(2);
        }
    }

    @Override
    public void visit(SimpleExpressionBlockNode node) {
        for (IRLExpression expression : node.getExpressionList()) {
            this.flops += multiplier * expression.getOperationCount();
        }
    }

    @Override
    public void visit(ComplexExpressionNode node) {
        this.flops += multiplier * node.getExpression().getOperationCount();
    }

    @Override
    public void visit(BranchNode node) {
        if (this.processor.getParent().getType().equals("GPU")) {
            multiplier *= WARP_PENALTY;
        }
    }

    @Override
    public void endVisit(BranchNode node) {
        if (this.processor.getParent().getType().equals("GPU")) {
            multiplier /= WARP_PENALTY;
        }
    }

    @Override
    public void visit(ForLoopNode node) {
        multiplier *= node.getNumIterations();
    }

    @Override
    public void endVisit(ForLoopNode node) {
        if (node.getNumIterations() == 0) {
            int a = 0;
        }
        multiplier = multiplier / node.getNumIterations();
    }

}
