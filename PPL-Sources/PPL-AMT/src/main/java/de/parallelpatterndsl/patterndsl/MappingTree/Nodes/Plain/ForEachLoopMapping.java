package de.parallelpatterndsl.patterndsl.MappingTree.Nodes.Plain;

import de.parallelpatterndsl.patterndsl.MappingTree.Nodes.MappingNode;
import de.parallelpatterndsl.patterndsl.MappingTree.Visitor.AMTVisitor;
import de.parallelpatterndsl.patterndsl.abstractPatternTree.DataElements.Data;
import de.parallelpatterndsl.patterndsl.abstractPatternTree.Nodes.Plain.ForEachLoopNode;
import de.parallelpatterndsl.patterndsl.hardwareDescription.Node;

import java.util.HashMap;
import java.util.Optional;

/**
 * Defines the for-each-loop node in abstract mapping trees.
 */
public class ForEachLoopMapping extends SerialNodeMapping {

    /**
     * The variable which defines the individual iterations of the loop.
     */
    private Data loopControlVariable;


    /**
     * The string used to differentiate different variables within the scope of the loop node.
     */
    private String generationRandomIndex;

    /**
     * The Expression defining the list to be iterated.
     */
    private ComplexExpressionMapping parsedList;

    public ForEachLoopMapping(Optional<MappingNode> parent, HashMap<String, Data> variableTable, ForEachLoopNode aptNode, Node target) {
        super(parent, variableTable, aptNode, target);
        loopControlVariable = aptNode.getLoopControlVariable();
        generationRandomIndex = aptNode.getGenerationRandomIndex();
    }

    public Data getLoopControlVariable() {
        return loopControlVariable;
    }

    public String getGenerationRandomIndex() {
        return generationRandomIndex;
    }

    public ComplexExpressionMapping getParsedList() {
        return parsedList;
    }

    public void setParsedList(ComplexExpressionMapping parsedList) {
        this.parsedList = parsedList;
    }

    public int getNumIterations() {
        return parsedList.getExpression().getShape().get(0);
    }

    public void setGenerationRandomIndex(String generationRandomIndex) {
        this.generationRandomIndex = generationRandomIndex;
    }

    /**
     * Visitor accept function.
     */
    public void accept(AMTVisitor visitor) {
        visitor.handle(this);
    }
}
