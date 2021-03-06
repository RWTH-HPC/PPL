package de.parallelpatterndsl.patterndsl.abstractPatternTree.Nodes.Plain;

import de.parallelpatterndsl.patterndsl.abstractPatternTree.Nodes.PatternNode;
import de.parallelpatterndsl.patterndsl.abstractPatternTree.Visitor.APTVisitor;
import de.parallelpatterndsl.patterndsl.abstractPatternTree.Visitor.CallCountResetter;
import de.parallelpatterndsl.patterndsl.abstractPatternTree.Visitor.ExtendedShapeAPTVisitor;

/**
 * Definition of a single case of a branch.
 */
public class BranchCaseNode extends PatternNode {

    /**
     * True, iff the Branch has a condition, which has to be fulfilled to execute it.
     */
    private boolean hasCondition;

    public BranchCaseNode(boolean hasCondition) {
        this.hasCondition = hasCondition;
    }

    public boolean isHasCondition() {
        return hasCondition;
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
