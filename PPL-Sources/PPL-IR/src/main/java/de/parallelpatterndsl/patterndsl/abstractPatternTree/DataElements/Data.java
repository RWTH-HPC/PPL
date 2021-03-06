package de.parallelpatterndsl.patterndsl.abstractPatternTree.DataElements;

import de.parallelpatterndsl.patterndsl.abstractPatternTree.DataTrace.DataTrace;

import java.util.ArrayList;

/**
 * Abstract definition of a data element within the parallel pattern diagram.
 */
public abstract class Data{

    /**
     * The identifier of the data element e.g. the name of a variable.
     */
    private String identifier;

    /**
     * The data type of the variable.
     */
    private final PrimitiveDataTypes typeName;

    /**
     * True, iff it does not need to be initialized.
     */
    private boolean isInitialized;

    /**
     * True, iff the data element is used as the result of the computation.
     */
    private boolean isReturnData;

    /**
     * The list of all data accesses and accessing nodes for this data element.
     */
    private DataTrace trace = new DataTrace(new ArrayList<>(), new ArrayList<>());

    /**
     * A list of indices suggesting when to create a copy of the data element, to resolve a read write conflict.
     * The value suggests before which data access the copy has to happen. e.g. a value of 3 means,
     * that the 3rd data access should be done on the copy of the data.
     */
    private ArrayList<Integer> copyIndices = new ArrayList<>();

    public Data(String identifier, PrimitiveDataTypes typeName, boolean isInitialized) {
        this.identifier = identifier;
        this.typeName = typeName;
        this.isInitialized = isInitialized;
        this.isReturnData = false;
    }

    public Data(String identifier, PrimitiveDataTypes typeName, boolean isInitialized, boolean isReturnData) {
        this.identifier = identifier;
        this.typeName = typeName;
        this.isInitialized = isInitialized;
        this.isReturnData = isReturnData;
    }

    /**
     * Returns true iff the data element is a function parameter.
     * @return
     */
    public boolean isParameter() {
        return isInitialized;
    }

    /**
     * Sets the value of initialized true, to help overviewing generated data elements.
     */
    public void setInitialized() {
        isInitialized = true;
    }

    /**
     * Returns an identifier for a data item.
     * @return
     */
    public String getIdentifier() {
        return identifier;
    }

    public void setIdentifier(String identifier) {
        this.identifier = identifier;
    }

    /**
     * The data type of the data item.
     * @return
     */
    public PrimitiveDataTypes getTypeName() {
        return typeName;
    }

    /**
     * Whether the data element is pre-initialized e.g. function parameters.
     * @return
     */
    public boolean isInitialized() {
        return isInitialized;
    }

    /**
     * The access trace of the of the given data element.
     * @return
     */
    public DataTrace getTrace() {
        return trace;
    }

    /**
     * The size in bytes of the data type.
     * @return
     */
    public int getSize() {
        return typeName.GetPrimitiveSize(typeName);
    }

    /**
     * stores whether the data element is a return element
     * @return
     */
    public boolean isReturnData() {
        return isReturnData;
    }


    /**
     * Returns the list of the indices from where to use the copy.
     * @return
     */
    public ArrayList<Integer> getCopyIndices() {
        return copyIndices;
    }

    /**
     * Adds a new index to the copyIndices list, signaling when to create a copy.
     * @param index
     */
    public void createCopy(int index) {
        this.copyIndices.add(index);
    }

    /**
     * Gives the number of bytes necessary to store the current data-element.0
     * @return
     */
    public abstract long getBytes();

    /**
     * returns a copy of the original data element for inlining purposes.
     * @param inlineIdentifier
     * @return
     */
    public abstract Data createInlineCopy(String inlineIdentifier);
}
