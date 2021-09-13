package de.parallelpatterndsl.patterndsl.abstractPatternTree.DataElements.DataAccess;

import de.parallelpatterndsl.patterndsl.abstractPatternTree.DataElements.Data;

import java.util.ArrayList;

/**
 * Definition of the data access within the dynamic programming pattern.
 */
public class DynamicProgrammingDataAccess extends DataAccess {
    /**
     * Linear offsets as part of the replication rule, for each individual dimension. Starting with the highest dimension.
     */
    private ArrayList<Integer> shiftOffsets;

    /**
     * The name of the INDEX variable as the base for the linear function.
     */
    private ArrayList<String> ruleBaseIndex;

    public DynamicProgrammingDataAccess(Data data, boolean isReadAccess, ArrayList<Integer> shiftOffsets, ArrayList<String> ruleBaseIndex) {
        super(data, isReadAccess);
        this.shiftOffsets = shiftOffsets;
        this.ruleBaseIndex = ruleBaseIndex;
    }

    public ArrayList<Integer> getShiftOffsets() {
        return shiftOffsets;
    }

    public ArrayList<String> getRuleBaseIndex() {
        return ruleBaseIndex;
    }
}
