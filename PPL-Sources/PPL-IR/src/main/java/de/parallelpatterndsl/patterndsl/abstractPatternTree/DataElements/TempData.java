package de.parallelpatterndsl.patterndsl.abstractPatternTree.DataElements;

/**
 * A temporary data value used to safely store sub-results from reductions.
 */
public class TempData extends PrimitiveData {
    public TempData(PrimitiveDataTypes typeName, String identifier) {
        super("tempData_" + identifier, typeName, false);
    }
}
