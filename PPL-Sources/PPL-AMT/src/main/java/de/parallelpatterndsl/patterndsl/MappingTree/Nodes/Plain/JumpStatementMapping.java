package de.parallelpatterndsl.patterndsl.MappingTree.Nodes.Plain;

import de.parallelpatterndsl.patterndsl.MappingTree.Nodes.MappingNode;
import de.parallelpatterndsl.patterndsl.MappingTree.Visitor.AMTVisitor;
import de.parallelpatterndsl.patterndsl.abstractPatternTree.DataElements.ArrayData;
import de.parallelpatterndsl.patterndsl.abstractPatternTree.DataElements.Data;
import de.parallelpatterndsl.patterndsl.abstractPatternTree.Nodes.PatternNode;
import de.parallelpatterndsl.patterndsl.hardwareDescription.Node;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.Optional;

/**
 * This class describes the target of a jump statement used for inlining calls within the APT.
 */
public class JumpStatementMapping extends SerialNodeMapping {

    /**
     * A list of array to be deallocated before leaving the scope-
     */
    private ArrayList<ArrayData> closingVars;

    /**
     * The operation executed before jumping to the defined label.
     */
    private ComplexExpressionMapping resultExpression;

    /**
     * Describes the label connecting the jump-Statement and label node.
     */
    private String label;

    /**
     * Stores the data element used to save the result of the inlined function
     */
    private Data outputData;


    public JumpStatementMapping(Optional<MappingNode> parent, HashMap<String, Data> variableTable, PatternNode node, ArrayList<ArrayData> closingVars, ComplexExpressionMapping resultExpression, String label, Data outputData, Node target) {
        super(parent, variableTable, node, target);
        this.closingVars = closingVars;
        this.resultExpression = resultExpression;
        this.label = label;
        this.outputData = outputData;
        this.children = new ArrayList<>();
        super.inputAccesses = new ArrayList<>(node.getInputAccesses());
        super.outputAccesses = new ArrayList<>(node.getOutputAccesses());
        super.setInputElements(new ArrayList<>(node.getInputElements()));
        super.setOutputElements(new ArrayList<>(node.getOutputElements()));
    }

    public JumpStatementMapping(Optional<MappingNode> parent, HashMap<String, Data> variableTable, PatternNode node, ArrayList<ArrayData> closingVars, String label, Data outputData, Node target) {
        super(parent, variableTable, node, target);
        this.closingVars = closingVars;
        this.label = label;
        this.outputData = outputData;
        this.children = new ArrayList<>();
        super.inputAccesses = new ArrayList<>(node.getInputAccesses());
        super.outputAccesses = new ArrayList<>(node.getOutputAccesses());
        super.setInputElements(new ArrayList<>(node.getInputElements()));
        super.setOutputElements(new ArrayList<>(node.getOutputElements()));
    }

    public ArrayList<ArrayData> getClosingVars() {
        return closingVars;
    }

    public ComplexExpressionMapping getResultExpression() {
        return resultExpression;
    }

    public String getLabel() {
        return label;
    }

    public void setResultExpression(ComplexExpressionMapping resultExpression) {
        this.resultExpression = resultExpression;
    }

    public Data getReturnOutputData() {
        return outputData;
    }

    /**
     * Visitor accept function.
     */
    public void accept(AMTVisitor visitor) {
        visitor.handle(this);
    }

}
