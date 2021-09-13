package de.parallelpatterndsl.patterndsl.patternSplits;

import de.parallelpatterndsl.patterndsl.abstractPatternTree.Nodes.Plain.ParallelCallNode;
import de.parallelpatterndsl.patterndsl.dataSplits.DataSplit;

import java.util.*;

/**
 * PatternSplit implementation for indicating the fuse (pipelines) of ParallelPatternSplit objects.
 */
public class FusedPatternSplit implements PatternSplit {

    private LinkedList<ParallelPatternSplit> jobs;

    public FusedPatternSplit(ParallelPatternSplit job) {
        this.jobs = new LinkedList<>();
        this.jobs.add(job);
    }

    @Override
    public ParallelCallNode getNode() {
        return this.jobs.peekFirst().getNode();
    }

    @Override
    public int[] getStartIndices() {
        return this.jobs.peekFirst().getStartIndices();
    }

    @Override
    public long[] getLengths() {
        return this.jobs.peekFirst().getLengths();
    }

    @Override
    public Set<DataSplit> getInputDataSplits() {
        return this.jobs.peekFirst().getInputDataSplits();
    }

    @Override
    public Set<DataSplit> getOutputDataSplits() {
        return this.jobs.peekLast().getOutputDataSplits();
    }

    public void append(ParallelPatternSplit job) {
        this.jobs.addLast(job);
    }

    public LinkedList<ParallelPatternSplit> getJobs() {
        return this.jobs;
    }

}
