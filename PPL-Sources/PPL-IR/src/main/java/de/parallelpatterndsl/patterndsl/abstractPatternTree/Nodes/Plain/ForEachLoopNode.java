package de.parallelpatterndsl.patterndsl.abstractPatternTree.Nodes.Plain;

import de.parallelpatterndsl.patterndsl.abstractPatternTree.DataElements.Data;
import de.parallelpatterndsl.patterndsl.abstractPatternTree.Nodes.PatternNode;
import de.parallelpatterndsl.patterndsl.abstractPatternTree.Visitor.APTVisitor;
import de.parallelpatterndsl.patterndsl.abstractPatternTree.Visitor.CallCountResetter;
import de.parallelpatterndsl.patterndsl.abstractPatternTree.Visitor.ExtendedShapeAPTVisitor;
import de.parallelpatterndsl.patterndsl.helperLibrary.DeepCopyHelper;

import java.util.ArrayList;

/**
 * Definition of a for-loop of the abstract pattern tree.
 * The first element in the children array is the assignment of the control variable.
 */
public class ForEachLoopNode extends LoopNode {

    /**
     * The variable which defines the individual iterations of the loop.
     */
    private Data loopControlVariable;


    /**
     * The string used to differentiate different variables within the scope of the loop node.
     */
    private String generationRandomIndex;


    public ForEachLoopNode(Data loopControlVariable) {
        this.loopControlVariable = loopControlVariable;
    }


    public Data getLoopControlVariable() {
        return loopControlVariable;
    }

    @Override
    public int getNumIterations() {
        ComplexExpressionNode definition = (ComplexExpressionNode) children.get(0);
        return definition.getExpression().getShape().get(0);
    }

    public String getGenerationRandomIndex() {
        return generationRandomIndex;
    }

    public void setGenerationRandomIndex(String generationRandomIndex) {
        this.generationRandomIndex = generationRandomIndex;
    }

    @Override
    public ForEachLoopNode deepCopy() {
        DeepCopyHelper.addScope(this.getVariableTable());
        ForEachLoopNode result = new ForEachLoopNode(DeepCopyHelper.currentScope().get(loopControlVariable.getIdentifier()));
        result.setVariableTable(DeepCopyHelper.currentScope());

        DeepCopyHelper.DataTraceUpdate(result);

        ArrayList<PatternNode> newChildren = new ArrayList<>();
        for (PatternNode node: this.getChildren()) {
            PatternNode newNode = node.deepCopy();
            newChildren.add(newNode);
            newNode.setParent(result);
        }

        result.setChildren(newChildren);

        DeepCopyHelper.removeScope();

        return result;
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
