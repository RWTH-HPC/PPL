package de.parallelpatterndsl.patterndsl.MappingTree.Nodes.DataControl;

import de.parallelpatterndsl.patterndsl.MappingTree.AbstractMappingTree;
import de.parallelpatterndsl.patterndsl.MappingTree.DataMovementGenerator.ParallelGroup;
import de.parallelpatterndsl.patterndsl.MappingTree.Nodes.MappingNode;
import de.parallelpatterndsl.patterndsl.MappingTree.Visitor.AMTVisitor;
import de.parallelpatterndsl.patterndsl.abstractPatternTree.DataElements.ArrayData;
import de.parallelpatterndsl.patterndsl.abstractPatternTree.DataElements.Data;
import de.parallelpatterndsl.patterndsl.abstractPatternTree.Nodes.PatternNode;
import de.parallelpatterndsl.patterndsl.hardwareDescription.Processor;

import java.util.*;

/**
 * class defining a barrier in an abstract mapping tree.
 */
public class BarrierMapping extends MappingNode {

    /**
     * The set of groups which need to wait for each other.
     */
    private Collection<ParallelGroup> barrier;

    /**
     * The set of processors which need to wait for each other.
     */
    private HashSet<Processor> barrierProc;

    /**
     * True, iff the barrier is defined by a set of call groups.
     */
    private boolean groupBased;

    public BarrierMapping(Optional<MappingNode> parent, HashMap<String, Data> variableTable, Collection<ParallelGroup> barrier) {
        super(parent, variableTable, new ArrayList<>(), new ArrayList<>(), new ArrayList<>(), new ArrayList<>());
        this.barrier = barrier;
        children = new ArrayList<>();
        groupBased = true;
        barrierProc = new HashSet<>();
        for (ParallelGroup group: barrier ) {
            barrierProc.addAll(group.getProcessors());
        }
    }

    public BarrierMapping(Optional<MappingNode> parent, HashMap<String, Data> variableTable, HashSet<Processor> barrierProc) {
        super(parent, variableTable, new ArrayList<>(), new ArrayList<>(), new ArrayList<>(), new ArrayList<>());
        this.barrierProc = barrierProc;
        children = new ArrayList<>();
        groupBased = false;
    }

    public Collection<ParallelGroup> getBarrier() {
        return barrier;
    }

    public HashSet<Processor> getBarrierProc() {
        return barrierProc;
    }

    public boolean isGroupBased() {
        return groupBased;
    }

    /**
     * Visitor accept function.
     */
    public void accept(AMTVisitor visitor) {
        visitor.handle(this);
    }
}
