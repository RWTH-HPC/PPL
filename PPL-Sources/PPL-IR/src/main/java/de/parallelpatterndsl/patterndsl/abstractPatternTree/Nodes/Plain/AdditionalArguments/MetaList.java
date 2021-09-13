package de.parallelpatterndsl.patterndsl.abstractPatternTree.Nodes.Plain.AdditionalArguments;

import java.util.ArrayList;
/**
 * Stores a set of values of meta information for the parallel call node.
 * @param <T>
 */
public class MetaList<T> extends AdditionalArguments {

    ArrayList<T> values;

    public ArrayList<T> getValues() {
        return values;
    }

    public MetaList(ArrayList<T> values) {
        this.values = values;
    }
}
