package de.parallelpatterndsl.patterndsl.abstractPatternTree.DataElements.DataAccess;

import de.parallelpatterndsl.patterndsl.abstractPatternTree.DataElements.Data;

/**
 * The definition of a parallel data access within a map pattern.
 */
public class MapDataAccess extends DataAccess {

    /**
     * Linear scaling factor as part of the replication rule.
     */
    private int scalingFactor;

    /**
     * Linear offset as part of the replication rule.
     */
    private int shiftOffset;

    public MapDataAccess(Data data, boolean isReadAccess, int scalingFactor, int shiftOffset) {
        super(data, isReadAccess);
        this.scalingFactor = scalingFactor;
        this.shiftOffset = shiftOffset;
    }

    public int getScalingFactor() {
        return scalingFactor;
    }

    public int getShiftOffset() {
        return shiftOffset;
    }


}
