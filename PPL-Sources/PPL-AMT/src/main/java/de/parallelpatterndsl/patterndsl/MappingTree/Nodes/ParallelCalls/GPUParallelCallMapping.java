package de.parallelpatterndsl.patterndsl.MappingTree.Nodes.ParallelCalls;

import de.parallelpatterndsl.patterndsl.MappingTree.AbstractMappingTree;
import de.parallelpatterndsl.patterndsl.MappingTree.GeneralDataPlacementFunctions.OffloadDataEncoding;
import de.parallelpatterndsl.patterndsl.MappingTree.Nodes.DataControl.AbstractDataMovementMapping;
import de.parallelpatterndsl.patterndsl.MappingTree.Nodes.DataControl.BarrierMapping;
import de.parallelpatterndsl.patterndsl.MappingTree.Nodes.DataControl.DataMovementMapping;
import de.parallelpatterndsl.patterndsl.MappingTree.Nodes.DataControl.DataPlacement;
import de.parallelpatterndsl.patterndsl.MappingTree.Nodes.Function.ParallelMapping;
import de.parallelpatterndsl.patterndsl.MappingTree.Nodes.MappingNode;
import de.parallelpatterndsl.patterndsl.MappingTree.Visitor.AMTVisitor;
import de.parallelpatterndsl.patterndsl.abstractPatternTree.DataElements.Data;
import de.parallelpatterndsl.patterndsl.abstractPatternTree.Nodes.Plain.CallNode;
import de.parallelpatterndsl.patterndsl.hardwareDescription.Processor;

import java.util.*;

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

    /**
     * Stores the encodings for the input data
     */
    private ArrayList<OffloadDataEncoding> inputDataEncodings;

    /**
     * Stores the encodings for the output data
     */
    private ArrayList<OffloadDataEncoding> outputDataEncodings;

    public GPUParallelCallMapping(Optional<MappingNode> parent, HashMap<String, Data> variableTable, CallNode aptNode, ArrayList<Long> startIndex, ArrayList<Long> numIterations, Processor executor, int numThreads, Optional<BarrierMapping> dynamicProgrammingBarrier, Set<AbstractDataMovementMapping> dynamicProgrammingdataTransfers, int numBlocks) {
        super(parent, variableTable, aptNode, startIndex, numIterations, executor, numThreads, dynamicProgrammingBarrier, dynamicProgrammingdataTransfers);
        this.numBlocks = numBlocks;
        this.threadsPerBlock = numThreads;
        inputDataEncodings = new ArrayList<>();
        outputDataEncodings = new ArrayList<>();
    }

    public int getNumBlocks() {
        return numBlocks;
    }

    public int getThreadsPerBlock() {
        return threadsPerBlock;
    }

    public void setNumBlocks(int numBlocks) {
        this.numBlocks = numBlocks;
    }

    public ArrayList<OffloadDataEncoding> getInputDataEncodings() {
        return inputDataEncodings;
    }

    public ArrayList<OffloadDataEncoding> getOutputDataEncodings() {
        return outputDataEncodings;
    }

    @Override
    public HashSet<DataPlacement> getNecessaryData() {
        ParallelMapping functionMapping = (ParallelMapping) AbstractMappingTree.getFunctionTable().get(getFunctionIdentifier());
        HashSet<DataPlacement> result = new HashSet<>(super.getNecessaryData());
        HashSet<DataPlacement> result2 = functionMapping.getNecessaryData();

        return result;
    }

    @Override
    public HashSet<DataPlacement> getOutputData() {
        ParallelMapping functionMapping = (ParallelMapping) AbstractMappingTree.getFunctionTable().get(getFunctionIdentifier());
        HashSet<DataPlacement> result = new HashSet<>(super.getOutputData());
        return result;
    }

    /**
     * Visitor accept function.
     */
    public void accept(AMTVisitor visitor) {
        visitor.handle(this);
    }
}
