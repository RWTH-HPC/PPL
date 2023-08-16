package de.parallelpatterndsl.patterndsl.MappingTree.Nodes.Plain;

import de.parallelpatterndsl.patterndsl.MappingTree.Nodes.DataControl.DataPlacement;
import de.parallelpatterndsl.patterndsl.MappingTree.Nodes.MappingNode;
import de.parallelpatterndsl.patterndsl.MappingTree.Visitor.AMTVisitor;
import de.parallelpatterndsl.patterndsl.abstractPatternTree.DataElements.Data;
import de.parallelpatterndsl.patterndsl.abstractPatternTree.Nodes.PatternNode;
import de.parallelpatterndsl.patterndsl.hardwareDescription.Node;

import java.util.HashMap;
import java.util.HashSet;
import java.util.Optional;

/**
 * Defines the return statement in abstract mapping trees.
 */
public class ReturnMapping extends SerialNodeMapping {

    /**
     * Stores the result of the return statement.
     */
    private Optional<ComplexExpressionMapping> result;


    public ReturnMapping(Optional<MappingNode> parent, HashMap<String, Data> variableTable, PatternNode node, Node target) {
        super(parent, variableTable, node, target);
    }

    public Optional<ComplexExpressionMapping> getResult() {
        return result;
    }

    public void setResult(ComplexExpressionMapping result) {
        this.result = Optional.of(result);
    }

    @Override
    public HashSet<DataPlacement> getNecessaryData() {
        if (result.isPresent()) {
            return result.get().getNecessaryData();
        }
        return new HashSet<>();
    }

    /**
     * Visitor accept function.
     */
    public void accept(AMTVisitor visitor) {
        visitor.handle(this);
    }
}
