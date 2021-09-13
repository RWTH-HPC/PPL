package de.parallelpatterndsl.patterndsl.abstractPatternTree.Nodes.Functions;

import de.parallelpatterndsl.patterndsl.abstractPatternTree.Visitor.APTVisitor;
import de.parallelpatterndsl.patterndsl.abstractPatternTree.Visitor.CallCountResetter;
import de.parallelpatterndsl.patterndsl.abstractPatternTree.Visitor.ExtendedShapeAPTVisitor;

/**
 * Definition of the Dynamic Programming pattern with in the abstract pattern tree.
 */
public class DynamicProgrammingNode extends ParallelNode {

    /**
     * The dimensionality of the underlying data structure.
     */
    private int dimension;

    public DynamicProgrammingNode(String identifier, int dimension) {
        super(identifier);
        this.dimension = dimension;
    }

    public int getDimension() {
        return dimension;
    }

    /**
     * Visitor accept function.
     */
    @Override
    public void accept(APTVisitor visitor) {
        visitor.handle(this);
    }

    public void accept(ExtendedShapeAPTVisitor visitor) {
        visitor.handle(this);
        CallCountResetter resetter = new CallCountResetter();
        this.accept(resetter);
    }
}
