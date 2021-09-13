package de.parallelpatterndsl.patterndsl.abstractPatternTree.Nodes.Plain;

import de.parallelpatterndsl.patterndsl.abstractPatternTree.Nodes.PatternNode;

/**
 * Abstract definition of a loop for the abstract pattern tree.
 */
public abstract class LoopNode extends PatternNode {

    public LoopNode() {
    }

    /**
     * returns the number of iteration done within the loop.
     * @return
     */
    public abstract int getNumIterations();
}
