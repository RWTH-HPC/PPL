package de.parallelpatterndsl.patterndsl.abstractPatternTree.Nodes.Plain.AdditionalArguments;

/**
 * Stores a single value of meta information for the parallel call node.
 * @param <T>
 */
public class MetaValue<T> extends AdditionalArguments {

    T value;

    public MetaValue(T value) {
        this.value = value;
    }

    public T getValue() {
        return value;
    }
}
