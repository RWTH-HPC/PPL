package de.parallelpatterndsl.patterndsl.abstractPatternTree.DataElements;

/**
 * An extension for the data element to interpret it as a function call.
 */
public class FunctionReturnData extends Data{

    public FunctionReturnData(String identifier, PrimitiveDataTypes typeName) {
        super(identifier, typeName, false);
    }

    @Override
    public long getBytes() {
        return 0;
    }

    public Data createInlineCopy(String inlineIdentifier) {
        return new FunctionReturnData(getIdentifier(), getTypeName());
    }

    @Override
    public Data deepCopy() {
        return new FunctionReturnData(getIdentifier(), getTypeName());
    }
}
