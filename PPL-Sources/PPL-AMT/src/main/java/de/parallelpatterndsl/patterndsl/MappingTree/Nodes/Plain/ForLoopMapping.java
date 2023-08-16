package de.parallelpatterndsl.patterndsl.MappingTree.Nodes.Plain;

import de.parallelpatterndsl.patterndsl.MappingTree.Nodes.MappingNode;
import de.parallelpatterndsl.patterndsl.MappingTree.Visitor.AMTVisitor;
import de.parallelpatterndsl.patterndsl.abstractPatternTree.DataElements.Data;
import de.parallelpatterndsl.patterndsl.abstractPatternTree.Nodes.Plain.ForLoopNode;
import de.parallelpatterndsl.patterndsl.hardwareDescription.Node;

import java.util.HashMap;
import java.util.Optional;

/**
 * Defines the for-loop node in abstract mapping trees.
 */
public class ForLoopMapping extends SerialNodeMapping {

    /**
     * The expression initializing the loop control variable.
     */
    private ComplexExpressionMapping initExpression;

    /**
     * The expression defining the boundary for the loop iteration.
     */
    private ComplexExpressionMapping controlExpression;

    /**
     * The expression defining the update rule for the loop iteration.
     */
    private ComplexExpressionMapping updateExpression;

    /**
     * The Variable used for the Loop iteration.
     */
    private Data loopControlVariable;

    public ForLoopMapping(Optional<MappingNode> parent, HashMap<String, Data> variableTable, ForLoopNode aptNode, Node target) {
        super(parent, variableTable, aptNode, target);
        loopControlVariable = aptNode.getLoopControlVariable();
    }

    public ComplexExpressionMapping getInitExpression() {
        return initExpression;
    }

    public ComplexExpressionMapping getControlExpression() {
        return controlExpression;
    }

    public ComplexExpressionMapping getUpdateExpression() {
        return updateExpression;
    }

    public Data getLoopControlVariable() {
        return loopControlVariable;
    }

    public void setInitExpression(ComplexExpressionMapping initExpression) {
        this.initExpression = initExpression;
    }

    public void setControlExpression(ComplexExpressionMapping controlExpression) {
        this.controlExpression = controlExpression;
    }

    public void setUpdateExpression(ComplexExpressionMapping updateExpression) {
        this.updateExpression = updateExpression;
    }



    /**
     * Visitor accept function.
     */
    public void accept(AMTVisitor visitor) {
        visitor.handle(this);
    }
}
