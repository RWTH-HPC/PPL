package de.parallelpatterndsl.patterndsl.abstractPatternTree.DataElements.DataAccess;

import de.parallelpatterndsl.patterndsl.abstractPatternTree.DataElements.Data;
import de.parallelpatterndsl.patterndsl.helperLibrary.DeepCopyHelper;

/**
 * The definition of a parallel data access within a reduction pattern.
 */
public class ReduceDataAccess extends DataAccess {

    /**
     * The number of recombinations that are done in a single reduction step.
     */
    private int width;

    public ReduceDataAccess(Data data, boolean isReadAccess) {
        super(data, isReadAccess);
    }

    public int getWidth() {
        return width;
    }

    public void setWidth(int width) {
        this.width = width;
    }

    /**
     * creates a copy of this data access, with a different data element
     * @param data
     * @return
     */
    @Override
    public DataAccess getInlineCopy(Data data) {
        ReduceDataAccess result = new ReduceDataAccess(data, super.isReadAccess());
        result.setWidth(this.width);
        return result;
    }

    @Override
    public DataAccess deepCopy() {
        ReduceDataAccess result = new ReduceDataAccess(DeepCopyHelper.currentScope().get(getData().getIdentifier()), isReadAccess());
        result.setWidth(width);
        return result;
    }
}
