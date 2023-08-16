package de.parallelpatterndsl.patterndsl.patternSplits;

import de.parallelpatterndsl.patterndsl.abstractPatternTree.DataElements.ArrayData;
import de.parallelpatterndsl.patterndsl.abstractPatternTree.DataElements.PrimitiveData;
import de.parallelpatterndsl.patterndsl.abstractPatternTree.Nodes.PatternNode;
import de.parallelpatterndsl.patterndsl.abstractPatternTree.Nodes.Plain.ReturnNode;
import de.parallelpatterndsl.patterndsl.dataSplits.DataSplit;
import de.parallelpatterndsl.patterndsl.dataSplits.DataSplitTable;

import java.util.HashSet;
import java.util.Set;

/**
 * PatternSplit implementation for PatternNode objects, which are not ParallelCallNode objects.
 */
public class SerialPatternSplit implements PatternSplit {

    private final PatternNode node;

    private final HashSet<DataSplit> inputDataSplits;

    private final HashSet<DataSplit> outputDataSplits;

    public SerialPatternSplit(PatternNode node) {
        this.node = node;

        this.inputDataSplits = new HashSet<>();
        this.outputDataSplits = new HashSet<>();

        node.getInputElements().forEach(d -> {
            if (d instanceof PrimitiveData) {
                this.inputDataSplits.add(DataSplitTable.get((PrimitiveData) d));
            } else if (d instanceof ArrayData) {
                ArrayData array = (ArrayData) d;
                if(DataSplitTable.get(array, 0, array.getShape().get(0) - 1) != null) {
                    this.inputDataSplits.addAll(DataSplitTable.get(array, 0, array.getShape().get(0) - 1));
                }
            }
        });

        node.getOutputElements().forEach(d -> {
            if (d instanceof PrimitiveData) {
                this.outputDataSplits.add(DataSplitTable.get((PrimitiveData) d));
            } else if (d instanceof ArrayData) {
                ArrayData array = (ArrayData) d;
                this.outputDataSplits.addAll(DataSplitTable.get(array, 0, array.getShape().get(0)));
            }
        });
    }

    @Override
    public PatternNode getNode() {
        return node;
    }

    @Override
    public long[] getStartIndices() {
        return new long[] {0};
    }

    @Override
    public long[] getLengths() {
        return new long[] {1};
    }

    @Override
    public Set<DataSplit> getInputDataSplits() {
        return inputDataSplits;
    }

    @Override
    public Set<DataSplit> getOutputDataSplits() {
        return outputDataSplits;
    }

}
