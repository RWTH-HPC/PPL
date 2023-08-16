package de.parallelpatterndsl.patterndsl.abstractPatternTree.DataElements.DataAccess;

import de.parallelpatterndsl.patterndsl.abstractPatternTree.DataElements.Data;
import de.parallelpatterndsl.patterndsl.helperLibrary.DeepCopyHelper;

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

    /**
     * creates a copy of this data access, with a different data element
     * @param data
     * @return
     */
    public DataAccess getInlineCopy(Data data) {
        return new DataAccess(data, this.isReadAccess);
    }

    /**
     * Creates a copy of a Data Access
     * @return
     */
    public DataAccess deepCopy() {
        return new DataAccess(DeepCopyHelper.currentScope().get(data.getIdentifier()), isReadAccess);
    }

}
