package de.parallelpatterndsl.patterndsl.dataSplits;

import de.parallelpatterndsl.patterndsl.abstractPatternTree.DataElements.ArrayData;
import de.parallelpatterndsl.patterndsl.dataSplits.DataSplit;

/**
 * The ArrayDataSplit class implements the DataSplit interface for ArrayData objects of Pattern-IRL.
 */
public class ArrayDataSplit implements DataSplit {

    private final ArrayData data;

    private final int startIndex;

    private final int length;

    public ArrayDataSplit(ArrayData data, int startIndex, int length) {
        this.data = data;
        this.startIndex = startIndex;
        this.length = length;
    }

    @Override
    public ArrayData getData() {
        return this.data;
    }

    @Override
    public long getBytes() {
        return (long) this.data.getBytes() / this.data.getShape().get(0) * length;
    }

}
