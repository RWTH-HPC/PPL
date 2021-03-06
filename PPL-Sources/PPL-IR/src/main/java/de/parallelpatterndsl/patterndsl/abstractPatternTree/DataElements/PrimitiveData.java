package de.parallelpatterndsl.patterndsl.abstractPatternTree.DataElements;

import de.se_rwth.commons.logging.Log;

/**
 *  An extension of the data element to interpret it as a primitive variable.
 */
public class PrimitiveData extends Data{

    public PrimitiveData(String identifier, PrimitiveDataTypes typeName, boolean isParameter) {
        super(identifier, typeName, isParameter);
    }

    public PrimitiveData(String identifier, PrimitiveDataTypes typeName, boolean isInitialized, boolean isReturnData) {
        super(identifier, typeName, isInitialized, isReturnData);
    }

    @Override
    public long getBytes() {
        if (this.isParameter()) {
            Log.error(this.getIdentifier() + " is a parameter and does not need memory space!");
            return 0;
        }
        return PrimitiveDataTypes.GetPrimitiveSize(this.getTypeName());
    }

    public Data createInlineCopy(String inlineIdentifier) {
        return new PrimitiveData(getIdentifier() + "_" + inlineIdentifier, getTypeName(), isInitialized(), isReturnData());
    }
}
