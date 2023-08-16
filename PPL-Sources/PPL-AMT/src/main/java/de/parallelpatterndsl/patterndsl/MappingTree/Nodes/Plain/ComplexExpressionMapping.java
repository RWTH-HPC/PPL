package de.parallelpatterndsl.patterndsl.MappingTree.Nodes.Plain;

import de.parallelpatterndsl.patterndsl.MappingTree.AbstractMappingTree;
import de.parallelpatterndsl.patterndsl.MappingTree.Nodes.DataControl.DataPlacement;
import de.parallelpatterndsl.patterndsl.MappingTree.Nodes.DataControl.EndPoint;
import de.parallelpatterndsl.patterndsl.MappingTree.Nodes.MappingNode;
import de.parallelpatterndsl.patterndsl.MappingTree.Visitor.AMTVisitor;
import de.parallelpatterndsl.patterndsl.abstractPatternTree.DataElements.ArrayData;
import de.parallelpatterndsl.patterndsl.abstractPatternTree.DataElements.Data;
import de.parallelpatterndsl.patterndsl.abstractPatternTree.DataElements.FunctionInlineData;
import de.parallelpatterndsl.patterndsl.abstractPatternTree.DataElements.PrimitiveData;
import de.parallelpatterndsl.patterndsl.abstractPatternTree.Nodes.Plain.ComplexExpressionNode;
import de.parallelpatterndsl.patterndsl.expressions.AssignmentExpression;
import de.parallelpatterndsl.patterndsl.expressions.IRLExpression;
import de.parallelpatterndsl.patterndsl.expressions.OperationExpression;
import de.parallelpatterndsl.patterndsl.expressions.Operator;
import de.parallelpatterndsl.patterndsl.hardwareDescription.Node;

import java.util.*;

/**
 * Defines a singular expression within the abstract mapping tree.
 */
public class ComplexExpressionMapping extends SerialNodeMapping {

    /**
     * The expression it contains.
     */
    private IRLExpression expression;

    public ComplexExpressionMapping(Optional<MappingNode> parent, HashMap<String, Data> variableTable, ComplexExpressionNode aptNode, Node target) {
        super(parent, variableTable, aptNode, target);
        expression = aptNode.getExpression();
    }

    public IRLExpression getExpression() {
        return expression;
    }

    public void setExpression(IRLExpression expression) {
        this.expression = expression;
    }

    @Override
    public HashSet<DataPlacement> getNecessaryData() {
        HashSet<DataPlacement> result = new HashSet<>();

        OperationExpression operationExpression;

        if (expression instanceof AssignmentExpression) {
            operationExpression = ((AssignmentExpression) expression).getRhsExpression();
        } else {
            operationExpression = (OperationExpression) expression;
        }

        HashSet<Data> elements = new HashSet<>();
        for (Data inputs: operationExpression.getOperands()) {
            if (inputs instanceof ArrayData || inputs instanceof PrimitiveData) {
                elements.add(inputs);
            } else if (inputs instanceof FunctionInlineData) {
                elements.addAll(((FunctionInlineData) inputs).getNestedData());
            }
        }


        for (Data inputs: elements) {
            ArrayList<EndPoint> partial = new ArrayList<>();
            EndPoint element;
            if (inputs instanceof ArrayData) {
                element = new EndPoint(AbstractMappingTree.getDefaultDevice(), 0, ((ArrayData) inputs).getShape().get(0),  new HashSet<>(), false);
            } else {
                element = new EndPoint(AbstractMappingTree.getDefaultDevice(), 0, 1, new HashSet<>(), false);
            }
            partial.add(element);
            DataPlacement placement = new DataPlacement(partial, inputs);
            result.add(placement);
        }
        return result;
    }

    public boolean isArrayInitializer() {
        if (expression instanceof AssignmentExpression) {
            if (((AssignmentExpression) expression).getRhsExpression().getOperators().size() != 0) {
                if (((AssignmentExpression) expression).getRhsExpression().getOperators().get(0) == Operator.LEFT_ARRAY_DEFINITION) {
                    return true;
                }
            } else if (!((AssignmentExpression) expression).getRhsExpression().getOperands().isEmpty()) {
                if (((AssignmentExpression) expression).getRhsExpression().getOperands().get(0) instanceof FunctionInlineData) {
                    if (((FunctionInlineData) ((AssignmentExpression) expression).getRhsExpression().getOperands().get(0)).getCall().getOperands().get(0).getIdentifier().equals("init_List")) {
                        return true;
                    } else if (((FunctionInlineData) ((AssignmentExpression) expression).getRhsExpression().getOperands().get(0)).getCall().getOperands().get(0).getIdentifier().equals("copy")) {
                        return true;
                    }
                }
            }
        }
        return false;
    }

    /**
     * Visitor accept function.
     */
    public void accept(AMTVisitor visitor) {
        visitor.handle(this);
    }
}
