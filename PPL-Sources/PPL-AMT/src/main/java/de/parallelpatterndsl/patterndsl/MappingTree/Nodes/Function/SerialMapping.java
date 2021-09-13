package de.parallelpatterndsl.patterndsl.MappingTree.Nodes.Function;

import de.parallelpatterndsl.patterndsl.MappingTree.AbstractMappingTree;
import de.parallelpatterndsl.patterndsl.MappingTree.Nodes.DataControl.DataPlacement;
import de.parallelpatterndsl.patterndsl.MappingTree.Nodes.DataControl.EndPoint;
import de.parallelpatterndsl.patterndsl.MappingTree.Nodes.FunctionMapping;
import de.parallelpatterndsl.patterndsl.MappingTree.Nodes.MappingNode;
import de.parallelpatterndsl.patterndsl.MappingTree.Visitor.AMTVisitor;
import de.parallelpatterndsl.patterndsl.abstractPatternTree.DataElements.ArrayData;
import de.parallelpatterndsl.patterndsl.abstractPatternTree.DataElements.Data;
import de.parallelpatterndsl.patterndsl.abstractPatternTree.DataElements.PrimitiveDataTypes;
import de.parallelpatterndsl.patterndsl.abstractPatternTree.Nodes.Functions.SerialNode;

import java.util.ArrayList;
import java.util.HashSet;
import java.util.Optional;

/**
 * Defines a serial function in an abstract mapping tree.
 */
public class SerialMapping extends FunctionMapping {

    /**
     * The primitive type of this functions return value.
     */
    private PrimitiveDataTypes returnType;

    /**
     * True, iff the function returns some kind of list.
     */
    private boolean isList;

    /**
     * The shape of the value returned by this function.
     */
    private ArrayList<Integer> shape;

    public SerialMapping(SerialNode aptNode) {
        super(aptNode);
        returnType = aptNode.getReturnType();
        isList = aptNode.isList();
        shape = aptNode.getShape();
    }

    public PrimitiveDataTypes getReturnType() {
        return returnType;
    }

    public boolean isList() {
        return isList;
    }

    public ArrayList<Integer> getShape() {
        return shape;
    }



    /**
     * Visitor accept function.
     */
    public void accept(AMTVisitor visitor) {
        visitor.handle(this);
    }
}
