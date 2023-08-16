package de.parallelpatterndsl.patterndsl.abstractPatternTree.Nodes.Plain;

import de.parallelpatterndsl.patterndsl.abstractPatternTree.Nodes.PatternNode;
import de.parallelpatterndsl.patterndsl.abstractPatternTree.Visitor.APTVisitor;
import de.parallelpatterndsl.patterndsl.abstractPatternTree.Visitor.CallCountResetter;
import de.parallelpatterndsl.patterndsl.abstractPatternTree.Visitor.ExtendedShapeAPTVisitor;
import de.parallelpatterndsl.patterndsl.helperLibrary.DeepCopyHelper;

import java.util.ArrayList;

/**
 * Definition of a branch in the abstract pattern tree.
 */
public class BranchNode extends PatternNode {

    /**
     * True, iff the branch and all of its children have been inlined into the main node.
     */
    private boolean inlined;

    public BranchNode() {
        inlined = false;
    }

    @Override
    public BranchNode deepCopy() {

        BranchNode result = new BranchNode();

        DeepCopyHelper.basicSetup(this, result);

        return result;
    }

    public boolean isInlined() {
        return inlined;
    }

    public void setInlined(boolean inlined) {
        this.inlined = inlined;
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
