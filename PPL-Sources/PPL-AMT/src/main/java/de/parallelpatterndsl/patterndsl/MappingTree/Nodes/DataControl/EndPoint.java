package de.parallelpatterndsl.patterndsl.MappingTree.Nodes.DataControl;

import de.parallelpatterndsl.patterndsl.MappingTree.DataMovementGenerator.ParallelGroup;
import de.parallelpatterndsl.patterndsl.hardwareDescription.Device;
import de.parallelpatterndsl.patterndsl.hardwareDescription.Processor;

import java.util.ArrayList;
import java.util.HashSet;
import java.util.Optional;

/**
 * This class describes the place the data currently resides.
 */
public class EndPoint {

    /**
     * The physical location of the data.
     */
    private Device location;

    /**
     * The first element to be send.
     */
    private long start;

    /**
     * The number of elements to be sent.
     */
    private long length;

    /**
     * Contains the parallel groups last accessing this endpoint.
     */
    private HashSet<ParallelGroup> parallelAccess;

    /**
     * True, iff the partial data was written to.
     */
    private boolean hasParallelWriteAccess;

    public EndPoint(Device location, long start, long length, HashSet<ParallelGroup> parallelAccess, boolean hasParallelWriteAccess) {
        this.location = location;
        this.start = start;
        this.length = length;
        this.parallelAccess = parallelAccess;
        this.hasParallelWriteAccess = hasParallelWriteAccess;
    }

    public Device getLocation() {
        return location;
    }

    public long getStart() {
        return start;
    }

    public long getLength() {
        return length;
    }

    public void setLocation(Device location) {
        this.location = location;
    }

    public HashSet<ParallelGroup> getParallelAccess() {
        return parallelAccess;
    }

    public boolean isHasParallelWriteAccess() {
        return hasParallelWriteAccess;
    }

    public void setHasParallelWriteAccess(boolean hasParallelWriteAccess) {
        this.hasParallelWriteAccess = hasParallelWriteAccess;
    }

    public EndPoint clone() {
        return new EndPoint(location, start, length, parallelAccess, hasParallelWriteAccess);
    }
}
