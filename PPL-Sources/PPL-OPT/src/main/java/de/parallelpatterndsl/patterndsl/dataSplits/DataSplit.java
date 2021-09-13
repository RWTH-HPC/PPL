package de.parallelpatterndsl.patterndsl.dataSplits;

import de.parallelpatterndsl.patterndsl.abstractPatternTree.DataElements.Data;

/**
 * Interface providing the abstraction of data splits.
 */
public interface DataSplit {

    /**
     * Returns the associated underlying data of the IRL.
     * @return Data
     */
    Data getData();

    /**
     * Returns the nubmer of bytes of the data split.
     * @return int.
     */
    long getBytes();

}
