package de.parallelpatterndsl.patterndsl.patternSplits;

import de.parallelpatterndsl.patterndsl.abstractPatternTree.Nodes.PatternNode;
import de.parallelpatterndsl.patterndsl.dataSplits.DataSplit;

import java.util.Set;

/**
 * Pattern split interface provides the basic abstraction for splitting PatternNode objects.
 */
public interface PatternSplit {

    /**
     * Returns the associated PatternNode of the split.
     * @return PatternNode
     */
    public PatternNode getNode();

    /**
     * Returns the start indices of the split in every dimension.
     * @return int[]
     */
    long[] getStartIndices();

    /**
     * Returns the lengths of the split in every dimension.
     * @return int[]
     */
    long[] getLengths();

    /**
     * Returns the accumulated set of input data splits.
     * @return data splits
     */
    Set<DataSplit> getInputDataSplits();

    /**
     * Returns the accumulated set of output data splits.
     * @return data splits
     */
    Set<DataSplit> getOutputDataSplits();

}
