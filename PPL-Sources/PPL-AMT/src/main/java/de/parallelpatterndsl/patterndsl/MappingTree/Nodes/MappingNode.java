package de.parallelpatterndsl.patterndsl.MappingTree.Nodes;

import de.parallelpatterndsl.patterndsl.MappingTree.AbstractMappingTree;
import de.parallelpatterndsl.patterndsl.MappingTree.Nodes.DataControl.DataPlacement;
import de.parallelpatterndsl.patterndsl.MappingTree.Nodes.DataControl.EndPoint;
import de.parallelpatterndsl.patterndsl.MappingTree.Visitor.AMTVisitor;
import de.parallelpatterndsl.patterndsl.abstractPatternTree.DataElements.ArrayData;
import de.parallelpatterndsl.patterndsl.abstractPatternTree.DataElements.Data;
import de.parallelpatterndsl.patterndsl.abstractPatternTree.DataElements.DataAccess.DataAccess;
import de.parallelpatterndsl.patterndsl.abstractPatternTree.Nodes.PatternNode;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.HashSet;
import java.util.Optional;

/**
 * The Basic node within the abstract mapping tree.
 */
public abstract class MappingNode {

    /**
     * A list of subtrees, that is spanned by the current pattern node.
     */
    protected ArrayList<MappingNode> children;

    /**
     * The parent node of the current node.
     */
    private Optional<MappingNode> parent;

    /**
     * Hash-map containing all variables available in the current scope.
     */
    private HashMap<String, Data> variableTable;

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

    public boolean hasParent() {
        return parent.isPresent();
    }

    public MappingNode getParent() {
        return parent.get();
    }

    public HashMap<String, Data> getVariableTable() {
        return variableTable;
    }

    public MappingNode(Optional<MappingNode> parent, HashMap<String, Data> variableTable, PatternNode node) {
        this.parent = parent;
        this.variableTable = variableTable;
        this.inputElements = node.getInputElements();
        this.outputElements = node.getOutputElements();
        this.inputAccesses = node.getInputAccesses();
        this.outputAccesses = node.getOutputAccesses();
    }

    public MappingNode(Optional<MappingNode> parent, HashMap<String, Data> variableTable, ArrayList<Data> inputElements, ArrayList<Data> outputElements, ArrayList<DataAccess> inputAccesses, ArrayList<DataAccess> outputAccesses) {
        this.parent = parent;
        this.variableTable = variableTable;
        this.inputElements = inputElements;
        this.outputElements = outputElements;
        this.inputAccesses = inputAccesses;
        this.outputAccesses = outputAccesses;
    }

    public ArrayList<MappingNode> getChildren() {
        return children;
    }

    public void setChildren(ArrayList<MappingNode> children) {
        this.children = children;
    }

    public void setParent(Optional<MappingNode> parent) {
        this.parent = parent;
    }

    public ArrayList<Data> getInputElements() {
        return inputElements;
    }

    public void setInputElements(ArrayList<Data> inputElements) {
        this.inputElements = inputElements;
    }

    public ArrayList<Data> getOutputElements() {
        return outputElements;
    }

    public void setOutputElements(ArrayList<Data> outputElements) {
        this.outputElements = outputElements;
    }

    public ArrayList<DataAccess> getInputAccesses() {
        return inputAccesses;
    }

    public void setInputAccesses(ArrayList<DataAccess> inputAccesses) {
        this.inputAccesses = inputAccesses;
    }

    public ArrayList<DataAccess> getOutputAccesses() {
        return outputAccesses;
    }

    public void setOutputAccesses(ArrayList<DataAccess> outputAccesses) {
        this.outputAccesses = outputAccesses;
    }


    public HashSet<DataPlacement> getNecessaryData() {
        HashSet<DataPlacement> result = new HashSet<>();

        for (Data inputs: getInputElements() ) {
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


    public HashSet<DataPlacement> getOutputData() {
        HashSet<DataPlacement> result = new HashSet<>();

        for (Data outputs: getOutputElements() ) {
            ArrayList<EndPoint> partial = new ArrayList<>();
            EndPoint element;
            if (outputs instanceof ArrayData) {
                element = new EndPoint(AbstractMappingTree.getDefaultDevice(), 0, ((ArrayData) outputs).getShape().get(0), new HashSet<>(), false);
            } else {
                element = new EndPoint(AbstractMappingTree.getDefaultDevice(), 0, 1, new HashSet<>(), false);
            }
            partial.add(element);
            DataPlacement placement = new DataPlacement(partial, outputs);
            result.add(placement);
        }
        return result;
    }

    /**
     * Visitor functions.
     *
     */
    public abstract void accept(AMTVisitor visitor) ;
}
