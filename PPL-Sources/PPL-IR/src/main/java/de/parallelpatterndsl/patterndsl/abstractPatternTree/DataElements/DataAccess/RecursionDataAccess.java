package de.parallelpatterndsl.patterndsl.abstractPatternTree.DataElements.DataAccess;

import de.parallelpatterndsl.patterndsl.abstractPatternTree.DataElements.Data;

/**
 * Definition of the data access within the recursive pattern.
 */
public class RecursionDataAccess extends DataAccess {
    public RecursionDataAccess(Data data, boolean isReadAccess) {
        super(data, isReadAccess);
    }
}
