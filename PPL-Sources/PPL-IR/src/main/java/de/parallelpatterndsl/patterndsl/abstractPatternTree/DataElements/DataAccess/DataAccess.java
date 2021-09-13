package de.parallelpatterndsl.patterndsl.abstractPatternTree.DataElements.DataAccess;

import de.parallelpatterndsl.patterndsl.abstractPatternTree.DataElements.Data;

import java.util.Iterator;
import java.util.function.Consumer;
/**
 * Class the that stores the information for a specific data access on a single data item.
 */
public class DataAccess{

    private final Data data;

    private final boolean isReadAccess;

    public DataAccess(Data data, boolean isReadAccess) {
        this.data = data;
        this.isReadAccess = isReadAccess;
    }


    /**
     * The data element which is accessed.
     * @return
     */
    public Data getData() {
        return data;
    }

    /**
     * Type of the data access.
     * @return
     */
    public boolean isReadAccess() {
        return isReadAccess;
    }

}
