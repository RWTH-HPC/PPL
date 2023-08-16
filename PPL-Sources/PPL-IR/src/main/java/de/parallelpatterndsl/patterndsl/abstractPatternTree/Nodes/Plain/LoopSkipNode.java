package de.parallelpatterndsl.patterndsl.abstractPatternTree.Nodes.Plain;

import de.parallelpatterndsl.patterndsl.abstractPatternTree.Nodes.PatternNode;
import de.parallelpatterndsl.patterndsl.abstractPatternTree.Visitor.APTVisitor;
import de.parallelpatterndsl.patterndsl.abstractPatternTree.Visitor.CallCountResetter;
import de.parallelpatterndsl.patterndsl.abstractPatternTree.Visitor.ExtendedShapeAPTVisitor;
import de.parallelpatterndsl.patterndsl.helperLibrary.DeepCopyHelper;

import java.util.ArrayList;

/**
 * Defines the continue and break key words
 */
public class LoopSkipNode extends PatternNode {

    /**
     * True, iff the key word is break, else the key word is continue.
     */
    private boolean isBreak;

    public LoopSkipNode(boolean isBreak) {
        this.isBreak = isBreak;
    }

    public boolean isBreak() {
        return isBreak;
    }

    @Override
    public LoopSkipNode deepCopy() {
        LoopSkipNode result = new LoopSkipNode(isBreak);

        DeepCopyHelper.basicSetup(this, result);

        return result;
    }

    @Override
    public boolean containsSynchronization() {
        return true;
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
