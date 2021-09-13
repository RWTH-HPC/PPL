package de.parallelpatterndsl.patterndsl.abstractPatternTree.DataTrace;

import de.parallelpatterndsl.patterndsl.abstractPatternTree.DataElements.DataAccess.DataAccess;
import de.parallelpatterndsl.patterndsl.abstractPatternTree.Nodes.PatternNode;
import org.javatuples.Pair;


import java.util.ArrayList;
import java.util.Set;

/**
 * Wrapper class that stores the access trace for a single data item.
 */
public class DataTrace {

    /**
     * The node that accesses the data element.
     */
    private ArrayList<PatternNode> accessingNodes;

    /**
     * The data access on the node.
     */
    private ArrayList<DataAccess> dataAccesses;

    public DataTrace(ArrayList<PatternNode> accessingNodes, ArrayList<DataAccess> dataAccesses) {
        this.accessingNodes = accessingNodes;
        this.dataAccesses = dataAccesses;
    }

    /**
     * A list nodes accessing the corresponding data element in order of their sequential execution.
     * @return
     */
    public ArrayList<PatternNode> getAccessingNodes() {
        return accessingNodes;
    }

    /**
     * A list data accesses corresponding to the data element in order of their sequential execution.
     * @return
     */
    public ArrayList<DataAccess> getDataAccesses() {
        return dataAccesses;
    }


    /**
     * Adds a new element to the data trace.
     * @param node
     * @param access
     */
    public void addTraceElement(PatternNode node, DataAccess access) {
        dataAccesses.add(access);
        accessingNodes.add(node);
    }



}
