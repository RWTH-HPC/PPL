package de.parallelpatterndsl.patterndsl.MappingTree.GPUMaximizer;

import de.parallelpatterndsl.patterndsl.MappingTree.AbstractMappingTree;
import de.parallelpatterndsl.patterndsl.MappingTree.Nodes.Function.RecursionMapping;
import de.parallelpatterndsl.patterndsl.MappingTree.Nodes.ParallelCalls.GPUParallelCallMapping;
import de.parallelpatterndsl.patterndsl.MappingTree.Nodes.ParallelCalls.ParallelCallMapping;
import de.parallelpatterndsl.patterndsl.MappingTree.Nodes.ParallelCalls.ReductionCallMapping;
import de.parallelpatterndsl.patterndsl.MappingTree.Visitor.ExtendedAMTShapeVisitor;

import java.util.ArrayList;

public class GPUMaximizer implements ExtendedAMTShapeVisitor {

    private AbstractMappingTree AMT;

    public GPUMaximizer(AbstractMappingTree AMT) {
        this.AMT = AMT;
    }

    public void maximize(){
        AMT.getRoot().accept(this.getRealThis());
    }

    @Override
    public void traverse(ParallelCallMapping node) {

    }

    @Override
    public void endVisit(GPUParallelCallMapping node) {
        int blocks = node.getNumBlocks();
        int threads = node.getThreadsPerBlock();
        ArrayList<Long> iterations = node.getNumIterations();

        blocks = (int) (iterations.get(0) /threads + 1);
        node.setNumBlocks(blocks);
    }

    @Override
    public void endVisit(ReductionCallMapping node) {
        if (node.getOnGPU() && !node.isOnlyCombiner()) {
            int blocks = node.getNumBlocks();
            int threads = node.getNumThreads();
            ArrayList<Long> iterations = node.getNumIterations();

            blocks = (int) (iterations.get(0) /threads + 1);
            node.setNumBlocks(blocks);
        }
    }


}
