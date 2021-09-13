package de.parallelpatterndsl.patterndsl.MappingTree.Nodes.ParallelCalls;

import de.parallelpatterndsl.patterndsl.MappingTree.Nodes.DataControl.DataPlacement;
import de.parallelpatterndsl.patterndsl.MappingTree.Nodes.MappingNode;
import de.parallelpatterndsl.patterndsl.MappingTree.Visitor.AMTVisitor;
import de.parallelpatterndsl.patterndsl.abstractPatternTree.DataElements.Data;
import de.parallelpatterndsl.patterndsl.abstractPatternTree.DataElements.DataAccess.DataAccess;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.HashSet;
import java.util.Optional;

/**
 * Defines a set of parallel calls on the same device that are executed in sequence.
 */
public class FusedParallelCallMapping extends MappingNode {
    public FusedParallelCallMapping(Optional<MappingNode> parent, HashMap<String, Data> variableTable) {
        super(parent, variableTable, new ArrayList<>(), new ArrayList<>(), new ArrayList<>(), new ArrayList<>());
    }


    @Override
    public ArrayList<Data> getInputElements() {
        ArrayList<Data> res = new ArrayList<>();

        for (MappingNode child: this.getChildren() ) {
            res.addAll(child.getInputElements());
        }

        return res;
    }

    @Override
    public ArrayList<Data> getOutputElements() {
        ArrayList<Data> res = new ArrayList<>();

        for (MappingNode child: this.getChildren() ) {
            res.addAll(child.getOutputElements());
        }

        return res;
    }

    @Override
    public ArrayList<DataAccess> getInputAccesses() {
        ArrayList<DataAccess> res = new ArrayList<>();

        for (MappingNode child: this.getChildren() ) {
            res.addAll(child.getInputAccesses());
        }

        return res;
    }

    @Override
    public ArrayList<DataAccess> getOutputAccesses() {
        ArrayList<DataAccess> res = new ArrayList<>();

        for (MappingNode child: this.getChildren() ) {
            res.addAll(child.getOutputAccesses());
        }

        return res;
    }

    @Override
    public HashSet<DataPlacement> getNecessaryData() {
        return children.get(0).getNecessaryData();
    }

    @Override
    public HashSet<DataPlacement> getOutputData() {
        HashSet<DataPlacement> result = new HashSet<>();

        for (MappingNode call: children ) {
            result.addAll(call.getOutputData());
        }
        return result;
    }

    /**
     * Visitor accept function.
     */
    public void accept(AMTVisitor visitor) {
        visitor.handle(this);
    }
}
