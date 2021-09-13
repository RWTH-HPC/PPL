package de.parallelpatterndsl.patterndsl.dataSplits;

import de.parallelpatterndsl.patterndsl.abstractPatternTree.DataElements.Data;
import de.parallelpatterndsl.patterndsl.abstractPatternTree.DataElements.PrimitiveData;
import de.parallelpatterndsl.patterndsl.dataSplits.DataSplit;

/**
 * The PrimitiveDataSplit class implements the DataSplit interface for PrimitiveData objects of Pattern-IRL.
 */
public class PrimitiveDataSplit implements DataSplit {

    private final PrimitiveData data;

    public PrimitiveDataSplit(PrimitiveData data) {
        this.data = data;
    }

    @Override
    public Data getData() {
        return data;
    }

    @Override
    public long getBytes() {
        return data.getBytes();
    }

}
