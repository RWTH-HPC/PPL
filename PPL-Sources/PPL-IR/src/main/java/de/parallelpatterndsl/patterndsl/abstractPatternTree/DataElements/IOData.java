package de.parallelpatterndsl.patterndsl.abstractPatternTree.DataElements;

public class IOData extends Data {

    /**
     * True, iff the value is printed.
     */
    private boolean isOutput;

    /**
     * True, iff the IO access targets a file.
     */
    private boolean hasFileAccess;

    public IOData(String identifier, PrimitiveDataTypes typeName, boolean isOutput, boolean hasFileAccess) {
        super(identifier, typeName, false);
        this.isOutput = isOutput;
        this.hasFileAccess = hasFileAccess;
    }

    @Override
    public long getBytes() {
        return 0;
    }

    public boolean isOutput() {
        return isOutput;
    }

    public boolean isHasFileAccess() {
        return hasFileAccess;
    }

    public Data createInlineCopy(String inlineIdentifier) {
        return new IOData(getIdentifier() + "_" + inlineIdentifier, getTypeName(), isOutput, isHasFileAccess());
    }
}
