package de.parallelpatterndsl.patterndsl.performance;

import de.parallelpatterndsl.patterndsl.abstractPatternTree.Nodes.Plain.ParallelCallNode;

public interface ParallelismEstimator {

    /**
     *
     * @param node
     * @return
     */
    long[] estimate(ParallelCallNode node);

}
