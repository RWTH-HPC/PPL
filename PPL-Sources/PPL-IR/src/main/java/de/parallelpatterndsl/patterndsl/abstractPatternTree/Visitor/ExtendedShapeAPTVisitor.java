package de.parallelpatterndsl.patterndsl.abstractPatternTree.Visitor;

import de.parallelpatterndsl.patterndsl.abstractPatternTree.AbstractPatternTree;
import de.parallelpatterndsl.patterndsl.abstractPatternTree.DataElements.ArrayData;
import de.parallelpatterndsl.patterndsl.abstractPatternTree.DataElements.Data;
import de.parallelpatterndsl.patterndsl.abstractPatternTree.Nodes.Functions.ParallelNode;
import de.parallelpatterndsl.patterndsl.abstractPatternTree.Nodes.Functions.SerialNode;
import de.parallelpatterndsl.patterndsl.abstractPatternTree.Nodes.Plain.CallNode;
import de.parallelpatterndsl.patterndsl.abstractPatternTree.Nodes.Plain.ComplexExpressionNode;
import de.parallelpatterndsl.patterndsl.abstractPatternTree.Nodes.Plain.ParallelCallNode;
import de.parallelpatterndsl.patterndsl.expressions.OperationExpression;
import de.parallelpatterndsl.patterndsl.helperLibrary.PredefinedFunctions;

import java.util.ArrayList;

/**
 * This class builds the interface for a visitor, which retains the shape of function arguments and transfers them to the function definition.
 * Thus, function parameters will know their current shape.
 *
 * IMPORTANT: DO NOT OVERWRITE THE HANDLE FUNCTION!!!!
 */
public interface ExtendedShapeAPTVisitor extends APTVisitor {

    @Override
    default public void handle(ParallelCallNode node){

        // Sets the shape of the function parameter.
        ParallelNode function = (ParallelNode) AbstractPatternTree.getFunctionTable().get(node.getFunctionIdentifier());

        ArrayList<ArrayList<Integer>> oldShapes = new ArrayList<>();
        ArrayList<ArrayList<Integer>> newShapes = new ArrayList<>();

        int i = 0;

        ArrayList<OperationExpression> arguments = node.getArgumentExpressions();

        for (Data parameter: function.getArgumentValues() ) {
            if (parameter instanceof ArrayData) {
                oldShapes.add(((ArrayData) parameter).getShape());
                ((ArrayData) parameter).setShape(new ArrayList<>(arguments.get(i).getShape()));
                if (!AbstractPatternTree.isExtendedVisitorFinished()) {
                    newShapes.add(((ArrayData) parameter).getShape());
                }
            }
            i++;
        }

        // handle return element
        if (function.getReturnElement() instanceof ArrayData ) {
            if (node.getChildren().get(0) instanceof ComplexExpressionNode) {
                oldShapes.add(((ArrayData) function.getReturnElement()).getShape());
                ((ArrayData) function.getReturnElement()).setShape(new ArrayList<>(((ComplexExpressionNode) node.getChildren().get(0)).getExpression().getShape()));
            }
            if (!AbstractPatternTree.isExtendedVisitorFinished()) {
                newShapes.add(new ArrayList<>(((ArrayData) function.getReturnElement()).getShape()));
            }
        }

        if (!AbstractPatternTree.isExtendedVisitorFinished()) {
            function.addParameterShapes(newShapes);
        }

        getRealThis().visit(node);
        getRealThis().traverse(node);
        getRealThis().endVisit(node);
        node.incrementCallCount();
        function.incrementCurrentCall();

        i = 0;

        // Resets the shape of the function parameter.
        for (Data parameter: function.getArgumentValues() ) {
            if (parameter instanceof ArrayData) {
                ((ArrayData) parameter).setShape(oldShapes.get(i));
                i++;
            }
        }




    }

    @Override
    default public void handle(CallNode node){

        if (PredefinedFunctions.contains(node.getFunctionIdentifier())) {
            return;
        }

        // Sets the shape of the function parameter.
        SerialNode function = (SerialNode) AbstractPatternTree.getFunctionTable().get(node.getFunctionIdentifier());

        ArrayList<ArrayList<Integer>> oldShapes = new ArrayList<>();

        ArrayList<ArrayList<Integer>> newShapes = new ArrayList<>();

        int i = 0;

        ArrayList<OperationExpression> arguments = node.getArgumentExpressions();

        for (Data parameter: function.getArgumentValues() ) {
            if (parameter instanceof ArrayData) {
                oldShapes.add(((ArrayData) parameter).getShape());
                ((ArrayData) parameter).setShape(arguments.get(i).getShape());
                if (!AbstractPatternTree.isExtendedVisitorFinished()) {
                    newShapes.add(arguments.get(i).getShape());
                }
            }
            i++;
        }

        if (!AbstractPatternTree.isExtendedVisitorFinished()) {
            function.addParameterShapes(newShapes);
        }

        getRealThis().visit(node);
        getRealThis().traverse(node);
        getRealThis().endVisit(node);
        node.incrementCallCount();
        function.incrementCurrentCall();

        i = 0;

        // Resets the shape of the function parameter.
        for (Data parameter: function.getArgumentValues() ) {
            if (parameter instanceof ArrayData) {
                ((ArrayData) parameter).setShape(oldShapes.get(i));
                i++;
            }
        }
    }

}
