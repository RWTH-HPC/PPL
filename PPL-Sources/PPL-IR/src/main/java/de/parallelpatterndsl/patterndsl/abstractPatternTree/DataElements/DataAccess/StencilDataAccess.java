package de.parallelpatterndsl.patterndsl.abstractPatternTree.DataElements.DataAccess;

import de.parallelpatterndsl.patterndsl.abstractPatternTree.DataElements.Data;
import de.parallelpatterndsl.patterndsl.helperLibrary.DeepCopyHelper;

import java.util.ArrayList;

/**
 * The definition of a parallel data access within a stencil pattern.
 */
public class StencilDataAccess extends DataAccess {

    /**
     * Linear scaling factors as part of the replication rule, for each individual dimension. Starting with the highest dimension.
     */
    private ArrayList<Integer> scalingFactors;

    /**
     * Linear offsets as part of the replication rule, for each individual dimension. Starting with the highest dimension.
     */
    private ArrayList<Integer> shiftOffsets;

    /**
     * The name of the INDEX variable as the base for the linear function.
     */
    private ArrayList<String> ruleBaseIndex;


    public StencilDataAccess(Data data, boolean isReadAccess, ArrayList<Integer> scalingFactors, ArrayList<Integer> shiftOffsets, ArrayList<String> ruleBaseIndex) {
        super(data, isReadAccess);
        this.scalingFactors = scalingFactors;
        this.shiftOffsets = shiftOffsets;
        this.ruleBaseIndex = ruleBaseIndex;
    }

    public ArrayList<Integer> getScalingFactors() {
        return scalingFactors;
    }

    public ArrayList<Integer> getShiftOffsets() {
        return shiftOffsets;
    }

    public ArrayList<String> getRuleBaseIndex() {
        return ruleBaseIndex;
    }

    /**
     * creates a copy of this data access, with a different data element
     * @param data
     * @return
     */
    @Override
    public DataAccess getInlineCopy(Data data) {
        return new StencilDataAccess(data, super.isReadAccess(), this.scalingFactors, this.shiftOffsets, this.ruleBaseIndex);
    }

    @Override
    public DataAccess deepCopy() {
        return new StencilDataAccess(DeepCopyHelper.currentScope().get(getData().getIdentifier()),isReadAccess(), new ArrayList<>(scalingFactors), new ArrayList<>(shiftOffsets), new ArrayList<>(ruleBaseIndex));
    }
}
