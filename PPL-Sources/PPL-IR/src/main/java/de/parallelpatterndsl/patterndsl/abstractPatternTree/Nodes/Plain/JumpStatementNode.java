package de.parallelpatterndsl.patterndsl.abstractPatternTree.Nodes.Plain;

import de.parallelpatterndsl.patterndsl.PatternTypes;
import de.parallelpatterndsl.patterndsl.abstractPatternTree.DataElements.ArrayData;
import de.parallelpatterndsl.patterndsl.abstractPatternTree.DataElements.Data;
import de.parallelpatterndsl.patterndsl.abstractPatternTree.DataElements.DataAccess.DataAccess;
import de.parallelpatterndsl.patterndsl.abstractPatternTree.Nodes.PatternNode;
import de.parallelpatterndsl.patterndsl.abstractPatternTree.Visitor.APTVisitor;
import de.parallelpatterndsl.patterndsl.abstractPatternTree.Visitor.CallCountResetter;
import de.parallelpatterndsl.patterndsl.abstractPatternTree.Visitor.ExtendedShapeAPTVisitor;
import de.parallelpatterndsl.patterndsl.expressions.OperationExpression;
import de.parallelpatterndsl.patterndsl.helperLibrary.DeepCopyHelper;

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
        ArrayList<DataAccess> accesses = resultExpression.getExpression().getDataAccesses(PatternTypes.SEQUENTIAL);
        for (DataAccess access : accesses) {
            if (access.isReadAccess()) {
                super.getInputAccesses().add(access);
                super.getInputElements().add(access.getData());
            } else {
                super.getOutputAccesses().add(access);
                super.getOutputElements().add(access.getData());
            }
        }
    }

    @Override
    public JumpStatementNode deepCopy() {

        ArrayList<ArrayData> newClosingVars = new ArrayList<>();

        for ( ArrayData data : closingVars) {
            newClosingVars.add((ArrayData) DeepCopyHelper.currentScope().get(data.getIdentifier()));
        }

        JumpStatementNode result = new JumpStatementNode(newClosingVars, resultExpression.deepCopy(), label, DeepCopyHelper.currentScope().get(outputData.getIdentifier()));

        DeepCopyHelper.basicSetup(this, result);

        return result;
    }

    @Override
    public boolean containsSynchronization() {
        return true;
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
