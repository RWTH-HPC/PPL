package de.parallelpatterndsl.patterndsl.abstractPatternTree.Nodes.Plain;

import de.parallelpatterndsl.patterndsl.abstractPatternTree.Visitor.APTVisitor;
import de.parallelpatterndsl.patterndsl.abstractPatternTree.Visitor.CallCountResetter;
import de.parallelpatterndsl.patterndsl.abstractPatternTree.Visitor.ExtendedShapeAPTVisitor;

/**
 * Class defining the While loop. Which executes as long as the condition resolves to true.
 * The condition is stored as an expression as the first child node.
 */
public class WhileLoopNode extends LoopNode{

    public WhileLoopNode() {
    }

    @Override
    public int getNumIterations() {
        return 1;
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
