package de.parallelpatterndsl.patterndsl.abstractPatternTree.Nodes.Functions;

import de.parallelpatterndsl.patterndsl.abstractPatternTree.Nodes.PatternNode;
import de.parallelpatterndsl.patterndsl.abstractPatternTree.Visitor.APTVisitor;
import de.parallelpatterndsl.patterndsl.abstractPatternTree.Visitor.CallCountResetter;
import de.parallelpatterndsl.patterndsl.abstractPatternTree.Visitor.ExtendedShapeAPTVisitor;
import de.parallelpatterndsl.patterndsl.helperLibrary.DeepCopyHelper;

import java.util.ArrayList;

/**
 * Definition of the stencil pattern of the abstract pattern tree.
 */
public class StencilNode extends ParallelNode {

    private int dimension;

    public StencilNode(String identifier, int dimension) {
        super(identifier);
        this.dimension = dimension;
    }

    public int getDimension() {
        return dimension;
    }

    public void setDimension(int dimension) {
        this.dimension = dimension;
    }

    @Override
    public StencilNode deepCopy() {
        StencilNode result = new StencilNode(getIdentifier(), dimension);

        DeepCopyHelper.basicPatternSetup(this, result);

        return result;
    }

    /**
     * Visitor accept function.
     */
    public void accept(APTVisitor visitor) {
        visitor.handle(this);
    }

    public void accept(ExtendedShapeAPTVisitor visitor) {
        visitor.handle(this);
        CallCountResetter resetter = new CallCountResetter();
        this.accept(resetter);
    }
}
