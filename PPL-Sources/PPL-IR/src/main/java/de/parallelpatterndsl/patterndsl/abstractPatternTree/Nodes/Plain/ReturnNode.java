package de.parallelpatterndsl.patterndsl.abstractPatternTree.Nodes.Plain;

import de.parallelpatterndsl.patterndsl.abstractPatternTree.DataElements.Data;
import de.parallelpatterndsl.patterndsl.abstractPatternTree.DataElements.PrimitiveData;
import de.parallelpatterndsl.patterndsl.abstractPatternTree.DataElements.PrimitiveDataTypes;
import de.parallelpatterndsl.patterndsl.abstractPatternTree.Nodes.FunctionNode;
import de.parallelpatterndsl.patterndsl.abstractPatternTree.Nodes.Functions.SerialNode;
import de.parallelpatterndsl.patterndsl.abstractPatternTree.Nodes.PatternNode;
import de.parallelpatterndsl.patterndsl.abstractPatternTree.Visitor.APTVisitor;
import de.parallelpatterndsl.patterndsl.abstractPatternTree.Visitor.CallCountResetter;
import de.parallelpatterndsl.patterndsl.abstractPatternTree.Visitor.ExtendedShapeAPTVisitor;
import de.parallelpatterndsl.patterndsl.helperLibrary.DeepCopyHelper;

import java.util.ArrayList;

public class ReturnNode extends PatternNode {

    public ReturnNode() {
    }

    @Override
    public ReturnNode deepCopy() {
        ReturnNode result = new ReturnNode();

        DeepCopyHelper.basicSetup(this, result);

        return result;
    }

    @Override
    public boolean containsSynchronization() {
        return true;
    }

    public PrimitiveDataTypes getFunctionType() {
        PatternNode node = this.getParent();
        while(!(node instanceof FunctionNode)) {
            node = node.getParent();
        }
        return ((FunctionNode) node).getReturnType();
    }

    public boolean doesReturnArray() {
        PatternNode node = this.getParent();
        while(!(node instanceof FunctionNode)) {
            node = node.getParent();
        }
        return ((FunctionNode) node).returnsArray();
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
