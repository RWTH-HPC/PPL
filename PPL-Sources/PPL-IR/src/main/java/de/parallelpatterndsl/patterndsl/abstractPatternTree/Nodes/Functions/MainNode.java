package de.parallelpatterndsl.patterndsl.abstractPatternTree.Nodes.Functions;

import de.parallelpatterndsl.patterndsl.abstractPatternTree.AbstractPatternTree;
import de.parallelpatterndsl.patterndsl.abstractPatternTree.DataElements.PrimitiveDataTypes;
import de.parallelpatterndsl.patterndsl.abstractPatternTree.Visitor.APTVisitor;
import de.parallelpatterndsl.patterndsl.abstractPatternTree.Visitor.CallCountResetter;
import de.parallelpatterndsl.patterndsl.abstractPatternTree.Visitor.ExtendedShapeAPTVisitor;

/**
 * Definition of the root node of the abstract pattern tree.
 */
public class MainNode extends SerialNode {

    public MainNode(String identifier, PrimitiveDataTypes returnType, boolean isList) {
        super(identifier, returnType, isList);
    }

    /**
     * Visitor accept function.
     */
    public void accept(APTVisitor visitor) {
        visitor.handle(this);
        if (visitor instanceof ExtendedShapeAPTVisitor) {
            CallCountResetter resetter = new CallCountResetter();
            this.accept(resetter);
            AbstractPatternTree.setExtendedVisitorFinished(true);
        }
    }



}
