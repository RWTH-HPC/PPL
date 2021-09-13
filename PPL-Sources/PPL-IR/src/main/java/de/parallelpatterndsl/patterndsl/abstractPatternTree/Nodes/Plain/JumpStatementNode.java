package de.parallelpatterndsl.patterndsl.abstractPatternTree.Nodes.Plain;

import de.parallelpatterndsl.patterndsl.abstractPatternTree.DataElements.ArrayData;
import de.parallelpatterndsl.patterndsl.abstractPatternTree.DataElements.Data;
import de.parallelpatterndsl.patterndsl.abstractPatternTree.Nodes.PatternNode;
import de.parallelpatterndsl.patterndsl.abstractPatternTree.Visitor.APTVisitor;
import de.parallelpatterndsl.patterndsl.abstractPatternTree.Visitor.CallCountResetter;
import de.parallelpatterndsl.patterndsl.abstractPatternTree.Visitor.ExtendedShapeAPTVisitor;
import de.parallelpatterndsl.patterndsl.expressions.OperationExpression;

import java.util.ArrayList;

public class JumpStatementNode extends PatternNode {

    /**
     * A list of array to be deallocated before leaving the scope-
     */
    private ArrayList<ArrayData> closingVars;

    /**
     * The operation executed before jumping to the defined label.
     */
    private ComplexExpressionNode resultExpression;

    /**
     * Describes the label connecting the jump-Statement and label node.
     */
    private String label;

    /**
     * Stores the data element used to save the result of the inlined function
     */
    private Data outputData;

    public String getLabel() {
        return label;
    }

    public void setLabel(String label) {
        this.label = label;
    }

    public ArrayList<ArrayData> getClosingVars() {
        return closingVars;
    }

    public void setClosingVars(ArrayList<ArrayData> closingVars) {
        this.closingVars = closingVars;
    }

    public ComplexExpressionNode getResultExpression() {
        return resultExpression;
    }

    public void setResultExpression(ComplexExpressionNode resultExpression) {
        this.resultExpression = resultExpression;
    }

    public Data getOutputData() {
        return outputData;
    }

    public JumpStatementNode(ArrayList<ArrayData> closingVars, ComplexExpressionNode resultExpression, String label, Data outputData) {
        this.closingVars = closingVars;
        this.resultExpression = resultExpression;
        this.label = label;
        this.outputData = outputData;
        children = new ArrayList<>();
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
