package de.parallelpatterndsl.patterndsl.dataSplits;

import de.parallelpatterndsl.patterndsl.abstractPatternTree.DataElements.Data;
import de.parallelpatterndsl.patterndsl.helperLibrary.RandomStringGenerator;

/**
 * Implements the DataSplit interface for temporary data, which is not specified in the original algorithm, but is necessary
 * when splitting recurrent patterns.
 */
public class TempDataSplit implements DataSplit {

    private final long bytes;

    private String identifier;

    public TempDataSplit(long bytes) {
        this.bytes = bytes;
        identifier = RandomStringGenerator.getAlphaNumericString();
    }

    @Override
    public Data getData() {
        return null;
    }

    @Override
    public long getBytes() {
        return bytes;
    }

    public String getIdentifier() {
        return identifier;
    }
}
