package de.parallelpatterndsl.patterndsl.MappingTree.Nodes.DataControl;

import de.parallelpatterndsl.patterndsl.MappingTree.AbstractMappingTree;
import de.parallelpatterndsl.patterndsl.MappingTree.Nodes.MappingNode;
import de.parallelpatterndsl.patterndsl.MappingTree.Visitor.AMTVisitor;
import de.parallelpatterndsl.patterndsl.abstractPatternTree.DataElements.ArrayData;
import de.parallelpatterndsl.patterndsl.abstractPatternTree.DataElements.Data;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.HashSet;
import java.util.Optional;

/**
 * Class defining data movement in an abstract mapping tree.
 */
public class DataMovementMapping extends AbstractDataMovementMapping {


    /**
     * Stores where the data comes from.
     */
    private HashSet<DataPlacement> Sender;

    /**
     * Stores where the data goes to.
     */
    private HashSet<DataPlacement> Receiver;


    public DataMovementMapping(Optional<MappingNode> parent, HashMap<String, Data> variableTable, HashSet<DataPlacement> sender, HashSet<DataPlacement> receiver) {
        super(parent, variableTable, new ArrayList<>(), new ArrayList<>(), new ArrayList<>(), new ArrayList<>());
        Sender = sender;
        Receiver = receiver;
        children = new ArrayList<>();
    }

    public HashSet<DataPlacement> getSender() {
        return Sender;
    }

    public HashSet<DataPlacement> getReceiver() {
        return Receiver;
    }

    @Override
    public HashSet<DataPlacement> getNecessaryData() {

        return Sender;
    }

    @Override
    public HashSet<DataPlacement> getOutputData() {

        return Receiver;
    }

    /**
     * Visitor accept function.
     */
    public void accept(AMTVisitor visitor) {
        visitor.handle(this);
    }
}
