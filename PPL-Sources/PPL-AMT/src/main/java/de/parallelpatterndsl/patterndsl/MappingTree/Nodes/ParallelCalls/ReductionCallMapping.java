package de.parallelpatterndsl.patterndsl.MappingTree.Nodes.ParallelCalls;

import de.parallelpatterndsl.patterndsl.MappingTree.Nodes.MappingNode;
import de.parallelpatterndsl.patterndsl.MappingTree.Visitor.AMTVisitor;
import de.parallelpatterndsl.patterndsl.abstractPatternTree.DataElements.Data;
import de.parallelpatterndsl.patterndsl.abstractPatternTree.DataElements.TempData;
import de.parallelpatterndsl.patterndsl.abstractPatternTree.Nodes.Plain.CallNode;
import de.parallelpatterndsl.patterndsl.hardwareDescription.Processor;

import java.util.*;

public class ReductionCallMapping extends ParallelCallMapping {

    /**
     * True, iff just the combiner function must be executed.
     */
    private boolean isOnlyCombiner;

    /**
     * Stores the temporary variables used as input.
     * If this set is empty the predefined input is assumed.
     */
    private Set<TempData> tempInput;

    /**
     * Stores the temporary variables used as output.
     * If this set is empty the predefined output is assumed.
     */
    private Set<TempData> tempOutput;

    /**
     * Stores the number of consecutive blocks on the GPU
     */
    private int numBlocks;

    /**
     * True, iff the reduction is executed on a GPU.
     */
    private boolean onGPU;

    public ReductionCallMapping(Optional<MappingNode> parent, HashMap<String, Data> variableTable, CallNode aptNode, ArrayList<Long> startIndex, ArrayList<Long> numIterations, Processor executor, int numThreads, boolean isOnlyCombiner, Set<TempData> tempInput, Set<TempData> tempOutput, int numBlocks, boolean onGPU) {
        super(parent, variableTable, aptNode, startIndex, numIterations, executor, numThreads, Optional.empty(), new HashSet<>());
        this.isOnlyCombiner = isOnlyCombiner;
        this.tempInput = tempInput;
        this.tempOutput = tempOutput;
        this.numBlocks = numBlocks;
        this.onGPU = onGPU;
    }

    public boolean isOnlyCombiner() {
        return isOnlyCombiner;
    }

    public Set<TempData> getTempInput() {
        return tempInput;
    }

    public Set<TempData> getTempOutput() {
        return tempOutput;
    }

    public int getNumBlocks() {
        return numBlocks;
    }

    public boolean getOnGPU() {
        return onGPU;
    }

    /**
     * Visitor accept function.
     */
    public void accept(AMTVisitor visitor) {
        visitor.handle(this);
    }
}
