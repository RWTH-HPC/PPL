package de.parallelpatterndsl.patterndsl.performance.simple;

import de.parallelpatterndsl.patterndsl.abstractPatternTree.AbstractPatternTree;
import de.parallelpatterndsl.patterndsl.abstractPatternTree.Nodes.FunctionNode;
import de.parallelpatterndsl.patterndsl.abstractPatternTree.Nodes.Functions.DynamicProgrammingNode;
import de.parallelpatterndsl.patterndsl.abstractPatternTree.Nodes.Functions.MapNode;
import de.parallelpatterndsl.patterndsl.abstractPatternTree.Nodes.Functions.ReduceNode;
import de.parallelpatterndsl.patterndsl.abstractPatternTree.Nodes.Functions.StencilNode;
import de.parallelpatterndsl.patterndsl.abstractPatternTree.Nodes.Plain.AdditionalArguments.MetaList;
import de.parallelpatterndsl.patterndsl.abstractPatternTree.Nodes.Plain.AdditionalArguments.MetaValue;
import de.parallelpatterndsl.patterndsl.abstractPatternTree.Nodes.Plain.ParallelCallNode;
import de.parallelpatterndsl.patterndsl.performance.ParallelismEstimator;

import java.util.Arrays;

/**
 * Retrieves the number of iterations of a ParallelCallNode object from the additional arguments of the IRL.
 */
public class SimpleParallelismEstimator implements ParallelismEstimator {

    /**
     *
     * @param node
     * @return
     */
    public long[] estimate(ParallelCallNode node) {
        FunctionNode pattern = AbstractPatternTree.getFunctionTable().get(node.getFunctionIdentifier());

        long[] lengths = null;
        if (pattern instanceof MapNode) {
            lengths = new long[1];
            lengths[0] = ((MetaValue<Long>) node.getAdditionalArguments().get(0)).getValue();
        } else if (pattern instanceof StencilNode) {
            MetaList<Long> metaValues = (MetaList<Long>) node.getAdditionalArguments().get(0);
            lengths = new long[metaValues.getValues().size()];
            for (int i = 0; i < lengths.length; i++) {
                lengths[i] = metaValues.getValues().get(i);
            }
        } else if (pattern instanceof ReduceNode) {
            MetaList<Long> metaValues = (MetaList<Long>) node.getAdditionalArguments().get(0);
            lengths = new long[1];
            lengths[0] = metaValues.getValues().get(0);
        } else if (pattern instanceof DynamicProgrammingNode) {
            lengths = new long[((MetaValue<Integer>) node.getAdditionalArguments().get(0)).getValue()];
            Arrays.fill(lengths, ((MetaValue<Long>) node.getAdditionalArguments().get(1)).getValue());
        }

        return lengths;
    }

}
