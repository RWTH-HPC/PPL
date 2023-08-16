package de.parallelpatterndsl.patterndsl.abstractPatternTree.Nodes.Plain;

import de.parallelpatterndsl.patterndsl.abstractPatternTree.Nodes.PatternNode;
import de.parallelpatterndsl.patterndsl.abstractPatternTree.Visitor.APTVisitor;
import de.parallelpatterndsl.patterndsl.abstractPatternTree.Visitor.CallCountResetter;
import de.parallelpatterndsl.patterndsl.abstractPatternTree.Visitor.ExtendedShapeAPTVisitor;
import de.parallelpatterndsl.patterndsl.helperLibrary.DeepCopyHelper;

import java.util.ArrayList;

/**
 * This class describes the target of a jump statement used for inlining calls within the APT.
 */
public class JumpLabelNode extends PatternNode {

    /**
     * Describes the label connecting the jump-Statement and label node.
     */
    private String label;

    public String getLabel() {
        return label;
    }


    public JumpLabelNode(String label) {
        this.label = label;
        children = new ArrayList<>();
    }

    @Override
    public JumpLabelNode deepCopy() {
        JumpLabelNode result = new JumpLabelNode(label);

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
