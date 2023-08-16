package de.parallelpatterndsl.patterndsl.abstractPatternTree.DataElements.DataAccess;

import de.parallelpatterndsl.patterndsl.abstractPatternTree.DataElements.Data;
import de.parallelpatterndsl.patterndsl.helperLibrary.DeepCopyHelper;

/**
 * Definition of the data access within the recursive pattern.
 */
public class RecursionDataAccess extends DataAccess {
    public RecursionDataAccess(Data data, boolean isReadAccess) {
        super(data, isReadAccess);
    }

    /**
     * creates a copy of this data access, with a different data element
     * @param data
     * @return
     */
    @Override
    public DataAccess getInlineCopy(Data data) {
        return new RecursionDataAccess(data, super.isReadAccess());
    }

    @Override
    public DataAccess deepCopy() {
        return new RecursionDataAccess(DeepCopyHelper.currentScope().get(getData().getIdentifier()), isReadAccess());
    }
}
