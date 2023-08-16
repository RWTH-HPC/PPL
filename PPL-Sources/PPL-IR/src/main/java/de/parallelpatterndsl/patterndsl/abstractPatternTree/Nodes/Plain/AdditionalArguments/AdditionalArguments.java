package de.parallelpatterndsl.patterndsl.abstractPatternTree.Nodes.Plain.AdditionalArguments;

/**
 * abstract class to stored different additional arguments from the front end as meta information for the parallel call node.
 */
public abstract class AdditionalArguments {

    /**
     * creates a deep copy of the additional argument
     * @return
     */
    public abstract AdditionalArguments deepCopy();
}
