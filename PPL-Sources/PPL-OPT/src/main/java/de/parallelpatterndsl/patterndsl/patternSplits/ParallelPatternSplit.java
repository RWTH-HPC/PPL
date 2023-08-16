package de.parallelpatterndsl.patterndsl.patternSplits;

import de.parallelpatterndsl.patterndsl.abstractPatternTree.Nodes.Plain.ParallelCallNode;
import de.parallelpatterndsl.patterndsl.dataSplits.DataSplit;

import java.util.Collection;
import java.util.HashSet;
import java.util.Set;

/**
 * The PatternSplit implementation for ParallelCallNode objects.
 */
public class ParallelPatternSplit implements PatternSplit {

    private final ParallelCallNode node;

    private long[] startIndices;

    private final long[] lengths;

    private final HashSet<DataSplit> inputDataSplits;

    private final HashSet<DataSplit> outputDataSplits;

    /**
     * Constructor of the ParallelPatternSplit object.
     * @param node - associated ParallelCallNode.
     * @param startIndices - start indices of split interval in every dimension.
     * @param lengths - length of split interval in every dimension.
     */
    public ParallelPatternSplit(ParallelCallNode node, long[] startIndices, long[] lengths) {
        this.node = node;
        this.startIndices = startIndices;
        this.lengths = lengths;

        this.inputDataSplits = new HashSet<>();
        this.outputDataSplits = new HashSet<>();
    }

    @Override
    public ParallelCallNode getNode() {
        return node;
    }

    @Override
    public long[] getStartIndices() {
        return startIndices;
    }

    @Override
    public long[] getLengths() {
        return lengths;
    }

    @Override
    public Set<DataSplit> getInputDataSplits() {
        return inputDataSplits;
    }

    @Override
    public Set<DataSplit> getOutputDataSplits() {
        return outputDataSplits;
    }

    public void addInputNetworkPackage(DataSplit pkg) {
        this.inputDataSplits.add(pkg);
    }

    public void addAllInputNetworkPackages(Collection<DataSplit> pkgs) {
        this.inputDataSplits.addAll(pkgs);
    }

    public void addOutputNetworkPackage(DataSplit pkg) {
        this.outputDataSplits.add(pkg);
    }

}
