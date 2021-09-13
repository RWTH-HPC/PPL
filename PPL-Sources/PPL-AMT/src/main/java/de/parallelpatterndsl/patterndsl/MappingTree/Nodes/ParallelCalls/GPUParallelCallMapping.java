package de.parallelpatterndsl.patterndsl.MappingTree.Nodes.ParallelCalls;

import de.parallelpatterndsl.patterndsl.MappingTree.Nodes.DataControl.BarrierMapping;
import de.parallelpatterndsl.patterndsl.MappingTree.Nodes.DataControl.DataMovementMapping;
import de.parallelpatterndsl.patterndsl.MappingTree.Nodes.MappingNode;
import de.parallelpatterndsl.patterndsl.MappingTree.Visitor.AMTVisitor;
import de.parallelpatterndsl.patterndsl.abstractPatternTree.DataElements.Data;
import de.parallelpatterndsl.patterndsl.abstractPatternTree.Nodes.Plain.CallNode;
import de.parallelpatterndsl.patterndsl.hardwareDescription.Processor;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.Optional;
import java.util.Set;

/**
 * Defines a parallel call on the GPU.
 * This class combines as many consecutive parallel calls as possible.
 */
public class GPUParallelCallMapping extends ParallelCallMapping {

    /**
     * Stores the number of consecutive blocks on the GPU
     */
    private int numBlocks;

    /**
     * Stores the number of threads for each block
     */
    private int threadsPerBlock;

    public GPUParallelCallMapping(Optional<MappingNode> parent, HashMap<String, Data> variableTable, CallNode aptNode, ArrayList<Long> startIndex, ArrayList<Long> numIterations, Processor executor, int numThreads, Optional<BarrierMapping> dynamicProgrammingBarrier, Set<DataMovementMapping> dynamicProgrammingdataTransfers, int numBlocks) {
        super(parent, variableTable, aptNode, startIndex, numIterations, executor, numThreads, dynamicProgrammingBarrier, dynamicProgrammingdataTransfers);
        this.numBlocks = numBlocks;
        this.threadsPerBlock = numThreads;
    }

    public int getNumBlocks() {
        return numBlocks;
    }

    public int getThreadsPerBlock() {
        return threadsPerBlock;
    }

    /**
     * Visitor accept function.
     */
    public void accept(AMTVisitor visitor) {
        visitor.handle(this);
    }
}
