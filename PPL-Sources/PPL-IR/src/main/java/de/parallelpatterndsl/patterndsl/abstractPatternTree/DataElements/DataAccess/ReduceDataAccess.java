package de.parallelpatterndsl.patterndsl.abstractPatternTree.DataElements.DataAccess;

import de.parallelpatterndsl.patterndsl.abstractPatternTree.DataElements.Data;

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
}
