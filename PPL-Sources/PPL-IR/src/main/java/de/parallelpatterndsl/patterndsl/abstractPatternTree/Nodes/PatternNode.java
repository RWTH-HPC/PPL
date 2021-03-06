package de.parallelpatterndsl.patterndsl.abstractPatternTree.Nodes;

import de.parallelpatterndsl.patterndsl.abstractPatternTree.AbstractPatternTree;
import de.parallelpatterndsl.patterndsl.abstractPatternTree.DataElements.Data;
import de.parallelpatterndsl.patterndsl.abstractPatternTree.DataElements.DataAccess.DataAccess;
import de.parallelpatterndsl.patterndsl.abstractPatternTree.Visitor.APTVisitor;

import java.util.ArrayList;
import java.util.HashMap;

/**
 * Class that describes the general information for every node within the abstract pattern tree.
 */
public abstract class PatternNode {

    /**
     * A list of subtrees, that is spanned by the current pattern node.
     */
    protected ArrayList<PatternNode> children;

    /**
     * The parent node of the current node.
     */
    private PatternNode parent;

    /**
     * Hash-map containing all variables available in the current scope.
     */
    private HashMap<String,Data> variableTable;

    /**
     * A list of all Variables read in this node.
     */
    private ArrayList<Data> inputElements;

    /**
     * A list of all variables changed in this node.
     */
    private ArrayList<Data> outputElements;

    /**
     * A list of the access patterns for all input elements.
     */
    protected ArrayList<DataAccess> inputAccesses;

    /**
     * A list of the access patterns for all output patterns.
     */
    protected ArrayList<DataAccess> outputAccesses;

    public ArrayList<PatternNode> getChildren() {
        return children;
    }

    public ArrayList<Data> getInputElements() {
        return inputElements;
    }

    public ArrayList<Data> getOutputElements() {
        return outputElements;
    }

    public ArrayList<DataAccess> getInputAccesses() {
        return inputAccesses;
    }

    public ArrayList<DataAccess> getOutputAccesses() {
        return outputAccesses;
    }

    public PatternNode getParent() {
        return parent;
    }

    public HashMap<String, Data> getVariableTable() {
        return variableTable;
    }

    public void setChildren(ArrayList<PatternNode> children) {
        this.children = children;
    }

    public void setParent(PatternNode parent) {
        this.parent = parent;
    }

    public void setVariableTable(HashMap<String, Data> variableTable) {
        this.variableTable = variableTable;
    }

    public void setInputElements(ArrayList<Data> inputElements) {
        this.inputElements = inputElements;
    }

    public void setOutputElements(ArrayList<Data> outputElements) {
        this.outputElements = outputElements;
    }

    public void setInputAccesses(ArrayList<DataAccess> inputAccesses) {
        this.inputAccesses = inputAccesses;
    }

    public void setOutputAccesses(ArrayList<DataAccess> outputAccesses) {
        this.outputAccesses = outputAccesses;
    }

    public PatternNode() {
        inputAccesses = new ArrayList<>();
        inputElements = new ArrayList<>();
        outputAccesses = new ArrayList<>();
        outputElements = new ArrayList<>();
    }

    /**
     * Visitor functions.
     *
     */
    public abstract void accept(APTVisitor visitor) ;

}
