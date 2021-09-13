package de.parallelpatterndsl.patterndsl.MappingTree.Visitor;

import de.parallelpatterndsl.patterndsl.MappingTree.AbstractMappingTree;
import de.parallelpatterndsl.patterndsl.MappingTree.Nodes.Function.ParallelMapping;
import de.parallelpatterndsl.patterndsl.MappingTree.Nodes.Function.SerialMapping;
import de.parallelpatterndsl.patterndsl.MappingTree.Nodes.ParallelCalls.GPUParallelCallMapping;
import de.parallelpatterndsl.patterndsl.MappingTree.Nodes.ParallelCalls.ParallelCallMapping;
import de.parallelpatterndsl.patterndsl.MappingTree.Nodes.ParallelCalls.ReductionCallMapping;
import de.parallelpatterndsl.patterndsl.MappingTree.Nodes.Plain.CallMapping;
import de.parallelpatterndsl.patterndsl.abstractPatternTree.AbstractPatternTree;
import de.parallelpatterndsl.patterndsl.abstractPatternTree.DataElements.ArrayData;
import de.parallelpatterndsl.patterndsl.abstractPatternTree.DataElements.Data;
import de.parallelpatterndsl.patterndsl.expressions.OperationExpression;

import java.util.ArrayList;

/**
 * This class builds the interface for a visitor, which retains the shape of function arguments and transfers them to the function definition.
 * Thus, function parameters will know their current shape.
 *
 * IMPORTANT: DO NOT OVERWRITE THE HANDLE FUNCTION!!!!
 */
public interface ExtendedAMTShapeVisitor extends AMTVisitor{

    @Override
    default public void handle(ParallelCallMapping node){
        parallelHandleWrapper(node);
    }

    @Override
    default public void handle(GPUParallelCallMapping node){
        parallelHandleWrapper(node);
    }

    @Override
    default public void handle(ReductionCallMapping node){
            parallelHandleWrapper(node);
    }

    @Override
    default public void handle(CallMapping node){

        if (node.getFunctionIdentifier().equals("init_List")) {
            return;
        }

        // Sets the shape of the function parameter.
        SerialMapping function = (SerialMapping) AbstractMappingTree.getFunctionTable().get(node.getFunctionIdentifier());

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


    public default void parallelHandleWrapper(ParallelCallMapping node) {
        // Sets the shape of the function parameter.
        ParallelMapping function = (ParallelMapping) AbstractMappingTree.getFunctionTable().get(node.getFunctionIdentifier());

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
            oldShapes.add(((ArrayData) function.getReturnElement()).getShape());
            ((ArrayData) function.getReturnElement()).setShape(new ArrayList<>((node.getDefinition()).getExpression().getShape()));

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
}
