package de.parallelpatterndsl.patterndsl.abstractPatternTree.DataElements.DataAccess;

import de.parallelpatterndsl.patterndsl.abstractPatternTree.DataElements.Data;
import de.parallelpatterndsl.patterndsl.helperLibrary.DeepCopyHelper;

/**
 * Class describing a data access where the data is accessed by IO-operations.
 * A read access in this case is accounted to a print operation and a write access to a read operation (when reading in files the target variable is changed!!!).
 */
public class IODataAccess extends DataAccess {
    public IODataAccess(Data data, boolean isReadAccess) {
        super(data, isReadAccess);
    }

    /**
     * creates a copy of this data access, with a different data element
     * @param data
     * @return
     */
    @Override
    public DataAccess getInlineCopy(Data data) {
        return new IODataAccess(data, super.isReadAccess());
    }

    @Override
    public DataAccess deepCopy() {
        return new IODataAccess(DeepCopyHelper.currentScope().get(getData().deepCopy()), isReadAccess());
    }
}
