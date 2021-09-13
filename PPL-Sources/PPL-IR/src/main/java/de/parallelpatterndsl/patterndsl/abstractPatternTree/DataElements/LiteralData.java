package de.parallelpatterndsl.patterndsl.abstractPatternTree.DataElements;

/**
 * An extension for the data element to cover literals within expressions.
 * @param <T>
 */
public class LiteralData <T> extends Data {

    /**
     * The Value of the literal.
     */
    private final T value;

    public LiteralData(String identifier, PrimitiveDataTypes typeName, T value) {
        super(identifier, typeName, false);
        this.value = value;
    }


    /**
     * The value of a given literal.
     * @return
     */
    public T getValue() {
        return value;
    }

    @Override
    public long getBytes() {
        return 0;
    }

    public Data createInlineCopy(String inlineIdentifier) {
        return new LiteralData<T>(getIdentifier(), getTypeName(), getValue());
    }
}
