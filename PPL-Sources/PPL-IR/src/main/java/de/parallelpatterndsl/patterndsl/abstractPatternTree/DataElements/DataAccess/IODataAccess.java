package de.parallelpatterndsl.patterndsl.abstractPatternTree.DataElements.DataAccess;

import de.parallelpatterndsl.patterndsl.abstractPatternTree.DataElements.Data;

/**
 * Class describing a data access where the data is accessed by IO-operations.
 * A read access in this case is accounted to a print operation and a write access to a read operation (when reading in files the target variable is changed!!!).
 */
public class IODataAccess extends DataAccess {
    public IODataAccess(Data data, boolean isReadAccess) {
        super(data, isReadAccess);
    }
}
