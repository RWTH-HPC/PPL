package de.parallelpatterndsl.patterndsl.abstractPatternTree.Nodes.Functions;

import de.parallelpatterndsl.patterndsl.abstractPatternTree.Nodes.PatternNode;
import de.parallelpatterndsl.patterndsl.abstractPatternTree.Visitor.APTVisitor;
import de.parallelpatterndsl.patterndsl.abstractPatternTree.Visitor.CallCountResetter;
import de.parallelpatterndsl.patterndsl.abstractPatternTree.Visitor.ExtendedShapeAPTVisitor;
import de.parallelpatterndsl.patterndsl.helperLibrary.DeepCopyHelper;

import java.util.ArrayList;

/**
 * Definition of the map Pattern in the abstract pattern tree.
 */
public class MapNode extends ParallelNode {


    public MapNode(String identifier) {
        super(identifier);
    }

    @Override
    public MapNode deepCopy() {
        MapNode result = new MapNode(getIdentifier());

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
