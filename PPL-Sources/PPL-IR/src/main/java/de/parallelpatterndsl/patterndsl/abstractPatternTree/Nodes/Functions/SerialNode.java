package de.parallelpatterndsl.patterndsl.abstractPatternTree.Nodes.Functions;

import de.parallelpatterndsl.patterndsl.abstractPatternTree.DataElements.PrimitiveDataTypes;
import de.parallelpatterndsl.patterndsl.abstractPatternTree.Nodes.FunctionNode;
import de.parallelpatterndsl.patterndsl.abstractPatternTree.Nodes.Plain.CallNode;
import de.parallelpatterndsl.patterndsl.abstractPatternTree.Nodes.Plain.ComplexExpressionNode;
import de.parallelpatterndsl.patterndsl.abstractPatternTree.Nodes.Plain.ParallelCallNode;
import de.parallelpatterndsl.patterndsl.abstractPatternTree.Nodes.Plain.ReturnNode;
import de.parallelpatterndsl.patterndsl.abstractPatternTree.Visitor.APTVisitor;
import de.parallelpatterndsl.patterndsl.abstractPatternTree.Visitor.CallCountResetter;
import de.parallelpatterndsl.patterndsl.abstractPatternTree.Visitor.ExtendedShapeAPTVisitor;
import de.se_rwth.commons.logging.Log;

import java.util.ArrayList;

/**
 * Definition of a serial function of the abstract pattern tree.
 */
public class SerialNode extends FunctionNode {

    /**
     * The primitive type of this functions return value.
     */
    private PrimitiveDataTypes returnType;

    /**
     * True, iff the function returns some kind of list.
     */
    private boolean isList;


    public SerialNode(String identifier, PrimitiveDataTypes returnType, boolean isList) {
        super(identifier);
        this.returnType = returnType;
        this.isList = isList;
    }

    public PrimitiveDataTypes getReturnType() {
        return returnType;
    }

    public boolean isList() {
        return isList;
    }

    /**
     * Returns the shape of the result. Returns an empty list, iff this function returns a scalar value.
     * @return
     */
    public ArrayList<Integer> getShape() {

        ArrayList<Integer> result = new ArrayList<>();

        if (isList) {
            returnVisitor visitor = new returnVisitor();
            this.accept(visitor);

            result = visitor.getShape();
        }

        return result;
    }

    /**
     * Visitor generating the shape of a function.
     */
    private class returnVisitor implements APTVisitor {

        private ArrayList<Integer> shape = new ArrayList<>();

        public ArrayList<Integer> getShape() {
            return shape;
        }

        @Override
        public void traverse(CallNode node) {}
        @Override
        public void traverse(ParallelCallNode node) {}

        @Override
        public void visit(ReturnNode node) {
            ArrayList<Integer> newShape = ((ComplexExpressionNode) node.getChildren().get(0)).getExpression().getShape();

            if (shape.size() != 0) {
                for (int i = 0; i < shape.size(); i++) {
                    if (shape.get(i) != newShape.get(i)) {
                        Log.error("Serial function not defined with a unique shape:  " + getIdentifier());
                        throw new RuntimeException("Critical error!");
                    }
                }
            }
            shape = newShape;
        }
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
